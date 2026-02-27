# NOTE:
# This module must NOT import config.py.
# All runtime behavior is driven by explicit config injection.
# 只在CPU

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
from .preprocess import (
    SNV,
    L2Normalize,
    MinMaxNormalize,
    augment_raw_spectrum,
    augment_norm_spectrum,
    sg_coeff,
)


class RamanDataset(Dataset):
    """
    层级自适应 Raman 数据集。

    目录结构:
        root / ... / ... / *.arc_data

    功能:
        - 自动识别目录深度
        - 生成各层级与 leaf 的标签映射
        - 维护父类 -> 子类映射，便于级联训练/推理
    """

    ROOT_TAG = "__root__"
    MISSING_TAG = "__missing__"

    def __init__(self, root_dir, augment=False, config=None):
        self.root_dir = root_dir
        self.augment = augment
        assert config is not None, "RamanDataset requires an explicit config"
        self.config = config

        # --------------------------------------------------
        # 主任务层级（兼容旧接口，默认 leaf）
        # --------------------------------------------------
        self.primary_level = getattr(self.config, "primary_level", None)

        # ==============================
        # 样本容器
        # ==============================
        self.samples = []          # file path
        self.level_labels = []     # 每个样本的多层标签（含 leaf）
        self.hier_names = []       # 每个样本的层级路径（前缀路径）

        # ==============================
        # 层级信息（扫描后填充）
        # ==============================
        self.level_names = []
        self.head_names = []
        self.head_name_to_idx = {}
        self.label_maps_by_level = []
        self.inv_label_maps_by_level = []
        self.class_names_by_level = []
        self.num_classes_by_level = {}

        # ==============================
        # 主任务兼容字段（用于旧脚本/评估）
        # ==============================
        self.labels = []
        self.class_names = []
        self.label_map = {}
        self.inv_label_map = {}

        # leaf 映射（便于外部读取）
        self.leaf_label_map = {}
        self.parent_level_name = {}
        self.parent_to_children = {}

        # ==============================
        # SG kernel 预生成（完全保留）
        # ==============================
        self.sg_smooth = torch.tensor(
            sg_coeff(self.config.win_smooth, 3, 0),
            dtype=torch.float32
        ).view(1, 1, -1)

        self.sg_d1 = torch.tensor(
            sg_coeff(self.config.win1, 3, 1),
            dtype=torch.float32
        ).view(1, 1, -1)

        # ==============================
        # 扫描数据（统一入口）
        # ==============================
        self._scan()

    # ======================================================
    # 统一层级扫描（核心）
    # ======================================================
    def _iter_leaf_dirs(self, root_dir):
        # 遍历所有包含 .arc_data 的叶子目录
        for root, dirs, files in os.walk(root_dir):
            dirs.sort()
            files.sort()
            arc_files = [f for f in files if f.lower().endswith(".arc_data")]
            if arc_files:
                yield root, arc_files

    def _parts_to_key(self, parts):
        # 统一用路径前缀字符串作为层级节点键
        if not parts:
            return self.ROOT_TAG
        return "/".join(parts)

    def _resolve_level_name(self, level_name):
        # 兼容旧字段：仍允许 leaf/level_x
        if level_name is None:
            return "leaf"
        if level_name in self.head_name_to_idx:
            return level_name
        if level_name == "leaf":
            return "leaf"
        if level_name.startswith("level_") and level_name in self.head_name_to_idx:
            return level_name
        return "leaf"

    def _scan(self):
        """
        扫描整个目录树:
            root / ... / ... / *.arc_data

        说明:
        - 扫描阶段与任务无关
        - 标签映射按层级节点（路径前缀）构建
        """
        leaf_records = []
        max_depth = 0

        for leaf_dir, arc_files in self._iter_leaf_dirs(self.root_dir):
            rel_dir = os.path.relpath(leaf_dir, self.root_dir)
            parts = [] if rel_dir == "." else rel_dir.split(os.sep)
            max_depth = max(max_depth, len(parts))
            for fname in arc_files:
                leaf_records.append((os.path.join(leaf_dir, fname), parts))

        # 层级名称（level_1 ... level_N）+ leaf
        self.level_names = [f"level_{i + 1}" for i in range(max_depth)]
        self.head_names = list(self.level_names) + ["leaf"]
        self.head_name_to_idx = {n: i for i, n in enumerate(self.head_names)}

        level_maps = [dict() for _ in range(max_depth)]
        leaf_map = {}

        # 建立层级节点映射
        for _, parts in leaf_records:
            leaf_key = self._parts_to_key(parts)
            if leaf_key not in leaf_map:
                leaf_map[leaf_key] = len(leaf_map)
            for i in range(len(parts)):
                key = self._parts_to_key(parts[: i + 1])
                if key not in level_maps[i]:
                    level_maps[i][key] = len(level_maps[i])

        # 保存映射与类名
        self.label_maps_by_level = level_maps + [leaf_map]
        self.leaf_label_map = leaf_map
        self.class_names_by_level = []
        self.inv_label_maps_by_level = []
        self.num_classes_by_level = {}

        for idx, name in enumerate(self.head_names):
            label_map = self.label_maps_by_level[idx]
            class_names = list(label_map.keys())
            inv_map = {i: n for n, i in label_map.items()}
            self.class_names_by_level.append(class_names)
            self.inv_label_maps_by_level.append(inv_map)
            self.num_classes_by_level[name] = len(class_names)

        # 构建父类 -> 子类映射（用于分层训练/推理）
        for idx, name in enumerate(self.head_names):
            if idx == 0:
                self.parent_level_name[name] = None
                self.parent_to_children[name] = {}
                continue

            parent_name = self.head_names[idx - 1]
            self.parent_level_name[name] = parent_name
            mapping = defaultdict(set)

            for _, parts in leaf_records:
                if name == "leaf":
                    if len(parts) < idx:
                        continue
                else:
                    if len(parts) < idx + 1:
                        continue

                parent_key = self._parts_to_key(parts[:idx])
                if name == "leaf":
                    child_key = self._parts_to_key(parts)
                else:
                    child_key = self._parts_to_key(parts[: idx + 1])

                parent_idx = self.label_maps_by_level[idx - 1][parent_key]
                if child_key not in self.label_maps_by_level[idx]:
                    continue
                child_idx = self.label_maps_by_level[idx][child_key]
                mapping[parent_idx].add(child_idx)

            self.parent_to_children[name] = {
                k: sorted(v) for k, v in mapping.items()
            }

        # 写入样本与标签
        for fpath, parts in leaf_records:
            labels = [-1] * len(self.head_names)
            hier = {n: None for n in self.head_names}

            for i in range(len(parts)):
                key = self._parts_to_key(parts[: i + 1])
                labels[i] = level_maps[i][key]
                hier[self.level_names[i]] = key

            leaf_key = self._parts_to_key(parts)
            labels[self.head_name_to_idx["leaf"]] = leaf_map[leaf_key]
            hier["leaf"] = leaf_key

            self.samples.append(fpath)
            self.level_labels.append(labels)
            self.hier_names.append(hier)

        # numpy 化，方便上层使用
        self.samples = np.array(self.samples)
        self.level_labels = np.array(self.level_labels, dtype=np.int64)
        self.hier_names = np.array(self.hier_names)

        # 绑定主任务（用于旧接口/评估）
        primary = self._resolve_level_name(self.primary_level)
        if primary not in self.head_name_to_idx:
            primary = "leaf"
        self.primary_level = primary
        primary_idx = self.head_name_to_idx[primary]

        if len(self.level_labels) > 0:
            self.labels = self.level_labels[:, primary_idx]
        else:
            self.labels = np.array([], dtype=np.int64)

        self.class_names = (
            self.class_names_by_level[primary_idx] if self.class_names_by_level else []
        )
        self.label_map = (
            self.label_maps_by_level[primary_idx] if self.label_maps_by_level else {}
        )
        self.inv_label_map = (
            self.inv_label_maps_by_level[primary_idx] if self.inv_label_maps_by_level else {}
        )
        self.num_classes = len(self.class_names)
        self.task_name = primary

    # ======================================================
    # Dataset 基本接口
    # ======================================================
    def __len__(self):
        return len(self.samples)

    # ======================================================
    # 读取 + 增强 + SG + 通道堆叠
    # ======================================================
    def __getitem__(self, idx):
        path = self.samples[idx]  # 路径
        labels = self.level_labels[idx]  # 多层标签
        hier = self.hier_names[idx]  # 层级
        # DataLoader default collate can't handle None in dict values.
        if isinstance(hier, dict):
            hier = {
                k: (v if v is not None else self.MISSING_TAG)
                for k, v in hier.items()
            }

        # ------------------------------
        # 读取光谱
        # ------------------------------
        data = np.loadtxt(path).astype(np.float32)
        raw_intensity = data[:, 1]

        # -------- RAW + pre-augment（强度空间）--------
        if self.augment and (not self.config.input_is_norm):
            raw_aug = augment_raw_spectrum(raw_intensity, self.config)
        else:
            raw_aug = raw_intensity.copy()

        x = raw_aug.copy()

        if not self.config.input_is_norm:
            if self.config.norm_method == "snv":
                x = SNV(x)

            elif self.config.norm_method == "l2":
                x = L2Normalize(x)

            elif self.config.norm_method == "minmax":
                x = MinMaxNormalize(x)
            else:
                raise ValueError(
                    f"Unknown norm_method: {self.config.norm_method}"
                )

        # -------- 形状空间 --------
        if self.augment:
            x = augment_norm_spectrum(x, self.config)

        # 构造 Tensor(cpu)
        signal = torch.tensor(
            x, dtype=torch.float32, device="cpu"
        ).view(1, 1, -1)

        # 通道构建
        base = signal[0, 0]
        if getattr(self.config, "snv_posneg_split", False):
            pos = torch.clamp(base, min=0.0)
            neg = torch.clamp(-base, min=0.0)
            channels = [pos, neg]
        else:
            channels = [base]

        # -----------------------------
        # smooth 通道
        # -----------------------------
        if self.config.smooth_use:
            smooth = F.conv1d(
                signal,
                self.sg_smooth,
                padding=self.config.win_smooth // 2
            )[0, 0]
            channels.append(smooth)


        # -----------------------------
        # 一阶导数
        # -----------------------------
        if self.config.d1_use:
            d1 = F.conv1d(
                signal,
                self.sg_d1,
                padding=self.config.win1 // 2
            )[0, 0]

            d1 = d1 / self.config.delta
            # ---- scale 对齐 ----
            scale = torch.max(torch.abs(d1)).clamp_min(1e-8)
            d1 = d1 / scale
            channels.append(d1)


        # -----------------------------
        # 最终检查 & stack
        # -----------------------------
        if len(channels) != self.config.in_channels:
            raise ValueError(
                f"Channel mismatch: built {len(channels)} channels, "
                f"but config.in_channels={self.config.in_channels}."
            )

        X = torch.stack(channels, dim=0)

        return X, labels, hier

    # ======================================================
    # --------- 额外接口（为 train / eval 预留）---------
    # ======================================================
    def get_hierarchy(self, idx):
        """
        返回某个样本的完整层级信息：
            {level_1: ..., level_2: ..., leaf: ...}

        不参与 __getitem__，避免影响 DataLoader
        """
        return self.hier_names[idx]

    def get_level_key(self, idx, level_name):
        level = self._resolve_level_name(level_name)
        hier = self.hier_names[idx]
        return hier.get(level)

    def get_parent_level(self, level_name):
        level = self._resolve_level_name(level_name)
        return self.parent_level_name.get(level)

    def get_split_key(self, idx, split_mode):
        """
        根据 split_mode 构造分组 key
        示例:
            split_mode = "level_2/leaf"
            -> (hier["level_2"], hier["leaf"])
        """
        keys = split_mode.split("/")
        if len(keys) == 1:
            return self.get_level_key(idx, keys[0])
        return tuple(self.get_level_key(idx, k) for k in keys)

    def encode_hierarchy(self, hier_list, device=None):
        """
        hier_list:
            - DataLoader default collate will turn list[dict] into dict[str, list]
            - Some callers may still pass list[dict] or numpy array of dict
        return: dict of tensors {level_1, level_2, ..., leaf}
        """
        # --------------------------------------------------
        # 统一输入格式
        # --------------------------------------------------
        if isinstance(hier_list, dict):
            names_by_level = {
                k: hier_list.get(k, []) for k in self.head_names
            }
        else:
            if hasattr(hier_list, "tolist"):
                hier_list = hier_list.tolist()

            if len(hier_list) == 0:
                names_by_level = {k: [] for k in self.head_names}
            elif isinstance(hier_list[0], dict):
                names_by_level = {
                    k: [h.get(k) for h in hier_list] for k in self.head_names
                }
            else:
                raise TypeError(
                    "encode_hierarchy expects a dict[str, list] (from DataLoader collate) "
                    "or a list/array of dicts from __getitem__."
                )

        # --------------------------------------------------
        # 名称 -> 稳定索引
        # --------------------------------------------------
        out = {}
        for idx, name in enumerate(self.head_names):
            label_map = self.label_maps_by_level[idx]
            labels = []
            for n in names_by_level[name]:
                if n is None:
                    labels.append(-1)
                else:
                    labels.append(label_map.get(n, -1))
            out[name] = torch.tensor(labels, dtype=torch.long)

        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}

        return out

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from raman.data.paths import resolve_dataset_stage
from raman.data.preprocess import (
    build_model_input,
    build_sg_kernels,
    load_arc_intensity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RamanDataset(Dataset):
    """
    层级自适应 Raman 数据集。

    目录结构示例：
        root / ... / ... / *.arc_data

    主要功能：
        - 自动识别目录深度
        - 生成各层级与 leaf 的标签映射
        - 维护父类 -> 子类映射，便于级联训练/推理
    """

    ROOT_TAG = "__root__"
    MISSING_TAG = "__missing__"

    def __init__(self, root_dir, augment=False, config=None):
        resolved_root = resolve_dataset_stage(
            root_dir,
            stage="train",
            project_root=PROJECT_ROOT,
            must_exist=True,
        )
        self.root_dir = os.fspath(resolved_root)
        self.augment = augment
        assert config is not None, "RamanDataset 必须显式传入 config"
        self.config = config

        # 样本容器
        self.samples = []          # 样本文件路径
        self.level_labels = []     # 每个样本的多层标签（含 leaf）
        self.hier_names = []       # 每个样本的层级路径（前缀路径）

        # 层级信息（扫描后填充）
        self.level_names = []
        self.head_names = []
        self.head_name_to_idx = {}
        self.label_maps_by_level = []
        self.inv_label_maps_by_level = []
        self.class_names_by_level = []
        self.num_classes_by_level = {}

        # leaf 映射（便于外部读取）
        self.leaf_label_map = {}
        self.parent_level_name = {}
        self.parent_to_children = {}

        # SG kernel 预生成
        self.sg_smooth, self.sg_d1 = build_sg_kernels(self.config, device="cpu")

        # 扫描数据
        self._scan()

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
        if level_name is None:
            return "leaf"
        if level_name not in self.head_name_to_idx:
            valid_levels = ", ".join(self.head_names)
            raise ValueError(
                f"未知层级名：{level_name}，可选值为：{valid_levels}"
            )
        return level_name

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]  # 路径
        labels = self.level_labels[idx]  # 多层标签
        hier = self.hier_names[idx]  # 层级
        # DataLoader 默认的 collate 不能直接处理 dict 中的 None 值。
        if isinstance(hier, dict):
            hier = {
                k: (v if v is not None else self.MISSING_TAG)
                for k, v in hier.items()
            }

        # 与 InputPreprocessor 复用同一套输入构建逻辑，避免训练/评估分支漂移。
        raw_intensity = load_arc_intensity(path)
        X = build_model_input(
            raw_intensity,
            config=self.config,
            sg_smooth=self.sg_smooth,
            sg_d1=self.sg_d1,
            device="cpu",
            augment=self.augment,
        )

        return X, labels, hier

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
        根据 split_mode 构造分组键。

        示例：
            split_mode = "level_2/leaf"
            -> (hier["level_2"], hier["leaf"])
        """
        keys = split_mode.split("/")
        if len(keys) == 1:
            return self.get_level_key(idx, keys[0])
        return tuple(self.get_level_key(idx, k) for k in keys)

    def encode_hierarchy(self, hier_list, device=None):
        """
        将层级名称结构编码成各层的整数标签张量。

        输入允许两种形式：
            - DataLoader 默认 collate 后得到的 dict[str, list]
            - 调用方直接传入的 list[dict] 或 numpy array[dict]

        返回：
            - dict[str, torch.Tensor]
            - 键为各层级名，值为对应的标签张量
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
                    "encode_hierarchy 只接受 DataLoader collate 后的 "
                    "dict[str, list]，或由 __getitem__ 返回的 dict 列表/数组。"
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

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_process.common import (
    build_wn_ref,
    preprocess_single_spectrum,
    read_arc_data,
    save_mean_plot,
)
from dataset_process.profiles import COMMON_BAD_BANDS

PACK_EXT = ".npz"

CUT_MIN = 600
CUT_MAX = 1800
TARGET_POINTS = 896
WN_REF = build_wn_ref(CUT_MIN, CUT_MAX, TARGET_POINTS)

ASLS_LAM = 3e5 # 改动
ASLS_P = 0.005 # 改动
ASLS_MAX_ITER = 15

MIN_SAMPLES_PER_CLASS = 8
NORM_METHOD = "snv"

PCA_ENABLED = True
PCA_COMPONENTS = 50
PCA_CENTER = True
PCA_OUTLIER_RATIO = 0.03


@dataclass(frozen=True)
class PipelineConfig:
    """集中管理离线预处理阶段的固定参数，便于 CLI 统一覆盖。"""
    cut_min: float = CUT_MIN
    cut_max: float = CUT_MAX
    target_points: int = TARGET_POINTS
    asls_lam: float = ASLS_LAM
    asls_p: float = ASLS_P
    asls_max_iter: int = ASLS_MAX_ITER
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS
    norm_method: str = NORM_METHOD
    pca_enabled: bool = PCA_ENABLED
    pca_components: float | int = PCA_COMPONENTS
    pca_center: bool = PCA_CENTER
    pca_outlier_ratio: float = PCA_OUTLIER_RATIO

    def build_wn_ref(self):
        """根据当前裁剪范围和目标点数生成统一插值坐标。"""
        return build_wn_ref(self.cut_min, self.cut_max, self.target_points)

DEFAULT_PIPELINE_CONFIG = PipelineConfig()

def resolve_path(base_dir, path_value):
    """将相对路径解析到当前数据集根目录下，统一得到绝对路径。"""
    return (Path(base_dir) / path_value).resolve()


def iter_arc_dirs(root_dir):
    """递归遍历目录树，只返回包含 .arc_data 文件的叶子目录。"""
    root_dir = os.fspath(root_dir)
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        arc_files = [name for name in files if name.lower().endswith(".arc_data")]
        if arc_files:
            yield Path(root), arc_files


def get_prefix(name):
    """统一按 letters_sign 规则提取类别前缀，兼容纯字母和字母后缀 +/-。"""
    matched = re.match(r"([A-Za-z]+)([+-])?", name)
    if not matched:
        return None
    return f"{matched.group(1)}{matched.group(2) or ''}"


def is_packed_path(path):
    """判断一个路径是否是可读取的打包数据文件。"""
    return os.path.isfile(path) and str(path).lower().endswith(PACK_EXT)


def write_arc_data(path, wn, sp, fmt="%.8f"):
    """把一条光谱写回两列文本格式，供后续训练和人工检查使用。"""
    arr = np.column_stack([wn, sp])
    np.savetxt(path, arr, fmt=[fmt, fmt])


def resolve_pipeline_config(pipeline_config=None):
    """返回离线预处理配置；未传入时使用库内默认配置。"""
    return pipeline_config or DEFAULT_PIPELINE_CONFIG


def _resolve_classify_target_dir(root_process_raw, rel_dir, leaf_name):
    """根据叶子目录名推断目标类别目录，统一处理顶层和多级目录。"""
    rel_parent = rel_dir.parent
    prefix = get_prefix(leaf_name)
    target_cls = prefix if prefix else leaf_name
    if rel_parent in (Path("."), Path("")):
        return root_process_raw / target_cls
    return root_process_raw / rel_parent / target_cls


def _resolve_group_figure_dir(root_figure, rel_dir):
    """为一个分组解析均值谱图输出目录，避免多处重复拼接父目录。"""
    rel_parent = rel_dir.parent
    if rel_parent in (Path("."), Path("")):
        return root_figure
    return root_figure / rel_parent


def _save_spectra_files(save_dir, filenames, wn_list, spectra_arr, fmt="%.3f"):
    """批量写出预处理后的光谱文件，统一文本精度和目录创建行为。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    for fname, wn_u, sp_u in zip(filenames, wn_list, spectra_arr):
        write_arc_data(save_dir / fname, wn_u, sp_u, fmt=fmt)


class PackedArcDataset:
    """从 dataset_init.npz 中按样本迭代恢复光谱内容。"""

    def __init__(self, npz_path):
        if not is_packed_path(npz_path):
            raise FileNotFoundError(f"Missing packed file: {npz_path}")
        data = np.load(npz_path)
        self.root_name = (
            str(data["root_name"][0]) if "root_name" in data else "dataset_init"
        )
        self.paths = data["paths"].tolist()
        self.offsets = data["offsets"]
        self.lengths = data["lengths"]
        self.wn_all = data["wn_all"]
        self.sp_all = data["sp_all"]

    def __len__(self):
        return len(self.paths)

    def get(self, index):
        start = int(self.offsets[index])
        length = int(self.lengths[index])
        end = start + length
        rel_path = self.paths[index]
        wn = self.wn_all[start:end]
        sp = self.sp_all[start:end]
        return rel_path, wn, sp

    def iter_samples(self):
        for index in range(len(self.paths)):
            yield self.get(index)


def resolve_init_input(base_dir, profile):
    """优先解析 dataset_init 目录，其次回退到打包后的 dataset_init.npz。"""
    root_init = resolve_path(base_dir, profile.root_init)
    root_init_pack = resolve_path(base_dir, profile.root_init_pack)

    if root_init.is_dir():
        return root_init
    if is_packed_path(root_init):
        return root_init
    if is_packed_path(root_init_pack):
        return root_init_pack

    raise FileNotFoundError(f"Missing input dir/file: {root_init}")


def iter_init_groups(input_path):
    """按叶子目录分组迭代原始样本，兼容目录输入和 npz 打包输入。"""
    input_path = Path(input_path)

    if input_path.is_dir():
        for leaf_dir, arc_files in iter_arc_dirs(input_path):
            rel_dir = leaf_dir.relative_to(input_path)
            samples = []
            for fname in arc_files:
                wn, sp = read_arc_data(leaf_dir / fname)
                samples.append((fname, wn, sp))
            yield rel_dir, leaf_dir.name, samples
        return

    packed = PackedArcDataset(input_path)
    grouped = {}

    for rel_path, wn, sp in packed.iter_samples():
        normalized_rel_path = rel_path.replace("\\", "/")
        rel_dir = Path(os.path.dirname(normalized_rel_path) or ".")
        group_key = rel_dir.as_posix()
        if group_key not in grouped:
            grouped[group_key] = {
                "rel_dir": rel_dir,
                "leaf_name": packed.root_name if rel_dir == Path(".") else rel_dir.name,
                "samples": [],
            }
        grouped[group_key]["samples"].append(
            (os.path.basename(normalized_rel_path), wn, sp)
        )

    for group in grouped.values():
        yield group["rel_dir"], group["leaf_name"], group["samples"]


def pack_dataset_init(input_dir, output_path, verbose=True):
    """把 dataset_init 下的散落光谱打包成一个 npz，便于迁移和归档。"""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    root_name = input_dir.resolve().name
    paths = []
    offsets = [0]
    wn_chunks = []
    sp_chunks = []

    for root, arc_files in iter_arc_dirs(input_dir):
        for fname in arc_files:
            full_path = root / fname
            wn, sp = read_arc_data(full_path)
            if wn.size == 0 or sp.size == 0:
                continue

            rel_path = full_path.relative_to(input_dir).as_posix()
            paths.append(rel_path)

            wn = wn.astype(np.float32)
            sp = sp.astype(np.float32)
            wn_chunks.append(wn)
            sp_chunks.append(sp)
            offsets.append(offsets[-1] + wn.size)

    if not paths:
        raise RuntimeError(f"No .arc_data files found under {input_dir}")

    wn_all = np.concatenate(wn_chunks, axis=0)
    sp_all = np.concatenate(sp_chunks, axis=0)
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    offsets = offsets[:-1]

    np.savez_compressed(
        output_path,
        root_name=np.asarray([root_name]),
        paths=np.asarray(paths),
        offsets=offsets,
        lengths=lengths,
        wn_all=wn_all,
        sp_all=sp_all,
    )

    if verbose:
        total = int(wn_all.size)
        print(f"[Pack] samples={len(paths)}, points={total}, saved={output_path}")


def unpack_dataset_init(npz_path, output_dir, verbose=True):
    """把 dataset_init.npz 恢复回目录树，便于重新检查和手工处理。"""
    npz_path = Path(npz_path)
    output_dir = Path(output_dir)
    packed = PackedArcDataset(npz_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    for rel_path, wn, sp in packed.iter_samples():
        out_path = output_dir / Path(rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_arc_data(out_path, wn, sp)
        restored += 1

    if verbose:
        print(f"[Unpack] samples={restored}, restored={output_dir}")


def classify_dataset(profile, base_dir):
    """将 dataset_init 重新归类到 dataset_train_raw，统一使用 letters_sign 前缀规则。"""
    base_dir = Path(base_dir)
    root_process_raw = resolve_path(base_dir, profile.root_process_raw)
    root_process_raw.mkdir(parents=True, exist_ok=True)

    input_path = resolve_init_input(base_dir, profile)

    copied = 0
    if Path(input_path).is_dir():
        for leaf_dir, arc_files in iter_arc_dirs(input_path):
            rel_dir = leaf_dir.relative_to(input_path)
            leaf_name = leaf_dir.name
            target_dir = _resolve_classify_target_dir(
                root_process_raw, rel_dir, leaf_name
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            for fname in arc_files:
                dst = target_dir / f"{leaf_name}_{fname}"
                shutil.copy(leaf_dir / fname, dst)
                copied += 1
    else:
        packed = PackedArcDataset(input_path)
        for rel_path, wn, sp in packed.iter_samples():
            normalized_rel_path = rel_path.replace("\\", "/")
            rel_dir = Path(os.path.dirname(normalized_rel_path) or ".")
            leaf_name = packed.root_name if rel_dir == Path(".") else rel_dir.name

            target_dir = _resolve_classify_target_dir(
                root_process_raw, rel_dir, leaf_name
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            fname = os.path.basename(normalized_rel_path)
            dst = target_dir / f"{leaf_name}_{fname}"
            write_arc_data(dst, wn, sp)
            copied += 1

    print(f"Stage 1 complete: copied {copied} files into {root_process_raw}")


def pca_reconstruct_and_error(spectra, n_components=0.95, center=True):
    """用 PCA 重构样本并返回逐样本误差，供训练集异常值过滤使用。"""
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim != 2 or spectra.shape[0] < 2:
        return spectra, 0, np.zeros((spectra.shape[0],), dtype=np.float32)

    n_samples = spectra.shape[0]
    mean = spectra.mean(axis=0, keepdims=True) if center else 0.0
    Xc = spectra - mean

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    if S.size == 0:
        return spectra, 0, np.zeros((spectra.shape[0],), dtype=np.float32)

    # 每个主成分对应的方差
    var = (S ** 2) / max(n_samples - 1, 1)
    total_var = float(var.sum())
    if total_var <= 0:
        return spectra.copy(), 0, np.zeros((n_samples,), dtype=np.float32)

    if isinstance(n_components, float) and 0 < n_components <= 1:
        cum_ratio = np.cumsum(var) / total_var
        k = int(np.searchsorted(cum_ratio, n_components) + 1)
    else:
        k = int(n_components)

    k = max(1, min(k, Vt.shape[0]))

    # X_hat = T_k P_k^T + mean
    # 其中 T_k = U[:, :k] * S[:k]
    X_hat = (U[:, :k] * S[:k]) @ Vt[:k, :]
    if center:
        X_hat = X_hat + mean

    errors = np.mean((spectra - X_hat) ** 2, axis=1).astype(np.float32)
    return X_hat, k, errors

def log_removed_samples(label, filenames, errors, threshold, log_path):
    """把 PCA 剔除掉的异常样本写入日志，方便后续追溯。"""
    if not filenames:
        return
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(
            f"[{label}] removed {len(filenames)} samples, "
            f"threshold={threshold:.6f}\n"
        )
        for fname, err in zip(filenames, errors):
            file.write(f"  {fname}\t{float(err):.6f}\n")

def preprocess_group_samples(
    samples,
    bad_bands,
    label_display,
    min_samples=None,
    log_path=None,
    apply_pca=None,
    pipeline_config=None,
):
    """对一个分组内的多条光谱做统一清洗，并按需执行 PCA 异常值过滤。"""
    cfg = resolve_pipeline_config(pipeline_config)

    if min_samples is None:
        min_samples = int(cfg.min_samples_per_class)
    if apply_pca is None:
        apply_pca = bool(cfg.pca_enabled)

    wn_ref = cfg.build_wn_ref()
    spectra, wn_list, filenames = [], [], []

    for fname, wn, sp in samples:
        if wn.size == 0 or sp.size == 0:
            continue

        wn_u, sp_u = preprocess_single_spectrum(
            wn,
            sp,
            cut_min=cfg.cut_min,
            cut_max=cfg.cut_max,
            wn_ref=wn_ref,
            bad_bands=bad_bands,
            asls_lam=cfg.asls_lam,
            asls_p=cfg.asls_p,
            asls_max_iter=cfg.asls_max_iter,
        )
        if wn_u is None:
            continue

        spectra.append(sp_u)
        wn_list.append(wn_u)
        filenames.append(fname)

    stats = {
        "input": len(samples),
        "valid_before_pca": len(spectra),
        "kept": len(spectra),
        "removed": 0,
        "pca_components": 0,
        "threshold": None,
        "skip_reason": None,
    }

    if len(spectra) < min_samples:
        stats["skip_reason"] = "too_few_preprocessed"
        return None, stats

    spectra_arr = np.vstack(spectra)

    if apply_pca and spectra_arr.shape[0] > 1:
        _, k, errors = pca_reconstruct_and_error(
            spectra_arr,
            n_components=cfg.pca_components,
            center=cfg.pca_center,
        )

        stats["pca_components"] = k

        if k > 0:
            ratio = float(np.clip(cfg.pca_outlier_ratio, 0.0, 1.0))
            if ratio <= 0.0:
                keep_mask = np.ones(len(errors), dtype=bool)
                threshold = float("inf")
            else:
                threshold = float(np.quantile(errors, 1.0 - ratio))
                keep_mask = errors <= threshold

            removed_mask = ~keep_mask
            stats["removed"] = int(removed_mask.sum())
            stats["threshold"] = threshold

            if stats["removed"] > 0 and log_path is not None:
                log_removed_samples(
                    label_display,
                    [f for f, rm in zip(filenames, removed_mask) if rm],
                    errors[removed_mask],
                    threshold,
                    log_path,
                )

            spectra_arr = spectra_arr[keep_mask]
            filenames = [f for f, keep in zip(filenames, keep_mask) if keep]
            wn_list = [wn for wn, keep in zip(wn_list, keep_mask) if keep]

    stats["kept"] = len(filenames)
    if len(filenames) < min_samples:
        stats["skip_reason"] = "too_few_after_pca"
        return None, stats

    return {
        "wn": wn_list[0],
        "spectra": spectra_arr,
        "filenames": filenames,
        "wn_list": wn_list,
    }, stats


def preprocess_train_dataset(profile, base_dir, pipeline_config=None):
    """从 dataset_train_raw 构建 dataset_train，并输出每类均值谱图和异常值日志。"""
    cfg = resolve_pipeline_config(pipeline_config)
    base_dir = Path(base_dir)
    root_process_raw = resolve_path(base_dir, profile.root_process_raw)
    root_process_clean = resolve_path(base_dir, profile.root_train_clean)
    root_figure = resolve_path(base_dir, profile.root_train_fig)
    log_path = resolve_path(base_dir, profile.log_name)

    if not root_process_raw.is_dir():
        raise FileNotFoundError(f"Missing input dir: {root_process_raw}")

    root_process_clean.mkdir(parents=True, exist_ok=True)
    root_figure.mkdir(parents=True, exist_ok=True)

    for cls_raw_dir, arc_files in iter_arc_dirs(root_process_raw):
        rel_dir = cls_raw_dir.relative_to(root_process_raw)
        label = rel_dir.as_posix() if rel_dir != Path(".") else root_process_raw.name
        label_display = label.replace("\\", "/")

        print(f"\n=== Processing: {label_display} ===")

        samples = []
        for fname in arc_files:
            wn, sp = read_arc_data(cls_raw_dir / fname)
            samples.append((fname, wn, sp))

        processed_group, stats = preprocess_group_samples(
            samples=samples,
            bad_bands=COMMON_BAD_BANDS,
            label_display=label_display,
            log_path=log_path,
            pipeline_config=cfg,
        )

        if stats["skip_reason"] == "too_few_preprocessed":
            print(
                f"  Skip: too few samples ({stats['valid_before_pca']}) in {label_display}"
            )
            continue

        if stats["pca_components"] > 0:
            print(
                f"  PCA outlier removal: k={stats['pca_components']}, "
                f"threshold={stats['threshold']:.6f}, removed={stats['removed']}"
            )

        if stats["skip_reason"] == "too_few_after_pca":
            print(f"  Skip: too few samples ({stats['kept']}) in {label_display}")
            continue

        spectra_arr = processed_group["spectra"]
        filenames = processed_group["filenames"]
        wn_list = processed_group["wn_list"]

        save_dir = root_process_clean / rel_dir
        _save_spectra_files(save_dir, filenames, wn_list, spectra_arr)

        fig_dir = _resolve_group_figure_dir(root_figure, rel_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_dir / f"{cls_raw_dir.name}.png"
        title = " - ".join(rel_dir.parts) + " (mean +/- std)"
        save_mean_plot(
            wn=wn_list[0],
            spectra=spectra_arr,
            out_path=fig_save_path,
            norm_method=cfg.norm_method,
            bad_bands=COMMON_BAD_BANDS,
            title=title,
        )

        print(f"  Mean spectrum saved: {fig_save_path}")

    print("\nTraining dataset preprocessing finished:")
    print(f"- Clean spectra: {root_process_clean}")
    print(f"- Mean plots: {root_figure}")


def preview_init_dataset(profile, base_dir, pipeline_config=None):
    """基于 dataset_init 生成预览图，不落盘清洗结果，适合先检查原始数据质量。"""
    cfg = resolve_pipeline_config(pipeline_config)
    base_dir = Path(base_dir)
    input_path = resolve_init_input(base_dir, profile)
    root_init_fig = resolve_path(base_dir, profile.root_init_fig)
    root_init_fig.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for rel_dir, leaf_name, samples in iter_init_groups(input_path):
        label = rel_dir.as_posix() if rel_dir != Path(".") else leaf_name
        label_display = label.replace("\\", "/")

        print(f"\n=== Preview: {label_display} ===")

        processed_group, stats = preprocess_group_samples(
            samples=samples,
            bad_bands=COMMON_BAD_BANDS,
            label_display=label_display,
            min_samples=1,
            log_path=None,
            apply_pca=False,
            pipeline_config=cfg,
        )

        if stats["skip_reason"] is not None:
            print(
                f"  Skip: no valid spectra after preprocessing "
                f"({stats['valid_before_pca']}/{stats['input']})"
            )
            skipped += 1
            continue

        if stats["pca_components"] > 0:
            print(
                f"  PCA outlier removal: k={stats['pca_components']}, "
                f"threshold={stats['threshold']:.6f}, removed={stats['removed']}"
            )

        fig_dir = _resolve_group_figure_dir(root_init_fig, rel_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_dir / f"{leaf_name}.png"

        title = " - ".join(rel_dir.parts) if rel_dir != Path(".") else leaf_name
        title = (
            f"{title} (mean +/- std, kept {stats['kept']}/{stats['input']})"
        )

        save_mean_plot(
            wn=processed_group["wn"],
            spectra=processed_group["spectra"],
            out_path=fig_save_path,
            norm_method=cfg.norm_method,
            bad_bands=COMMON_BAD_BANDS,
            title=title,
        )

        print(f"  Mean spectrum saved: {fig_save_path}")
        generated += 1

    print("\nDataset init preview finished:")
    print(f"- Mean plots: {root_init_fig}")
    print(f"- Generated={generated}, Skipped={skipped}")


def preprocess_test_dataset(
    profile,
    base_dir,
    input_dir=None,
    output_dir=None,
    pipeline_config=None,
):
    """从测试原始目录构建 dataset_test，并输出每个文件夹的均值谱图。"""
    cfg = resolve_pipeline_config(pipeline_config)
    wn_ref = cfg.build_wn_ref()
    base_dir = Path(base_dir)
    input_dir = (
        Path(input_dir)
        if input_dir is not None
        else resolve_path(base_dir, profile.root_test_raw)
    )
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_path(base_dir, profile.root_test_clean)
    )
    root_test_fig = resolve_path(base_dir, profile.root_test_fig)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    root_test_fig.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errored = 0

    for root, arc_files in iter_arc_dirs(input_dir):
        spectra = []
        wn_list = []

        for fname in arc_files:
            in_path = root / fname
            rel_dir = in_path.parent.relative_to(input_dir)
            out_dir = output_dir / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / fname

            try:
                wn, sp = read_arc_data(in_path)
                wn_u, sp_u = preprocess_single_spectrum(
                    wn,
                    sp,
                    cut_min=cfg.cut_min,
                    cut_max=cfg.cut_max,
                    wn_ref=wn_ref,
                    bad_bands=COMMON_BAD_BANDS,
                    asls_lam=cfg.asls_lam,
                    asls_p=cfg.asls_p,
                    asls_max_iter=cfg.asls_max_iter,
                )
                if wn_u is None:
                    rel_path = in_path.relative_to(input_dir).as_posix()
                    print(f"[SKIP] {rel_path} (empty after cut)")
                    skipped += 1
                    continue

                write_arc_data(out_path, wn_u, sp_u, fmt="%.3f")

                spectra.append(sp_u)
                wn_list.append(wn_u)
                processed += 1
            except Exception as exc:
                rel_path = in_path.relative_to(input_dir).as_posix()
                print(f"[ERROR] {rel_path}: {exc}")
                errored += 1

        if spectra:
            spectra_arr = np.vstack(spectra)
            wn_ref = wn_list[0]

            rel_dir = root.relative_to(input_dir)
            fig_dir = _resolve_group_figure_dir(root_test_fig, rel_dir)
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / f"{root.name}.png"

            save_mean_plot(
                wn=wn_ref,
                spectra=spectra_arr,
                out_path=fig_path,
                norm_method=cfg.norm_method,
                bad_bands=COMMON_BAD_BANDS,
                title=f"{rel_dir.as_posix()} (mean +/- std)",
            )

            print(f"  Mean spectrum saved: {fig_path}")

    print(
        "Test dataset preprocessing finished. "
        f"Processed={processed}, Skipped={skipped}, Error={errored}"
    )


def compute_totals(node):
    """递归回填每个目录节点的总样本数。"""
    total = node.get("__count__", 0)
    for name, child in node.items():
        if name.startswith("__"):
            continue
        total += compute_totals(child)
    node["__total__"] = total
    return total


def build_tree(root_dir):
    """把目录树转成带计数的嵌套字典，供 count 子命令打印。"""
    tree = {}
    for leaf_dir, arc_files in iter_arc_dirs(root_dir):
        rel_dir = Path(leaf_dir).relative_to(root_dir)
        parts = [] if rel_dir == Path(".") else rel_dir.parts

        node = tree
        for part in parts:
            node = node.setdefault(part, {})

        node["__count__"] = node.get("__count__", 0) + len(arc_files)

    compute_totals(tree)
    return tree


def count_dataset(root_dir):
    """统计一个数据目录下各层文件数，并返回树形结构。"""
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {root_dir}")

    tree = build_tree(root_dir)
    total_files = tree.get("__total__", 0)
    return tree, total_files


def print_tree(node, level=0, name=None):
    """按缩进样式打印统计树，便于终端查看目录层级分布。"""
    indent = "  " * level
    if name is not None:
        count = node.get("__count__", 0)
        total = node.get("__total__", 0)
        children = [key for key in node.keys() if not key.startswith("__")]
        if children:
            if count > 0:
                print(f"{indent}{name}: {count} 个文件 (含子目录总计 {total})")
            else:
                print(f"{indent}{name}: 总计 {total} 个文件")
        else:
            print(f"{indent}{name}: {count} 个文件")

    for child_name in sorted(key for key in node.keys() if not key.startswith("__")):
        print_tree(node[child_name], level + 1, child_name)


def print_results(tree, total_files):
    """统一打印 count 子命令的统计结果摘要。"""
    print("\n================ 数据集统计 ================\n")
    print(f"总文件数: {total_files}\n")

    root_count = tree.get("__count__", 0)
    if root_count:
        root_total = tree.get("__total__", 0)
        print(f"[根目录] {root_count} 个文件 (含子目录总计 {root_total})\n")

    for top_name in sorted(key for key in tree.keys() if not key.startswith("__")):
        print_tree(tree[top_name], 0, top_name)
        print("")

    print("============================================\n")

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raman.data.archive import (
    PackedArcDataset,
    iter_arc_dirs,
    iter_init_groups,
    resolve_init_input,
    resolve_path,
)
from raman.data.offline import (
    preprocess_single_spectrum,
    save_mean_plot,
)
from raman.data.profiles import COMMON_BAD_BANDS
from raman.data.spectrum import (
    build_wn_ref,
    read_arc_data,
    write_arc_data,
)

CUT_MIN = 600
CUT_MAX = 1800
TARGET_POINTS = 896

ASLS_LAM = 3e5
ASLS_P = 0.005
ASLS_MAX_ITER = 15

MIN_SAMPLES_PER_CLASS = 8
NORM_METHOD = "snv"

PCA_ENABLED = True
PCA_COMPONENTS = 50
PCA_CENTER = True
PCA_OUTLIER_RATIO = 0.03

@dataclass(frozen=True)
class PipelineConfig:
    """集中管理离线预处理阶段的固定参数，便于 CLI 统一覆盖"""
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
        """根据当前裁剪范围和目标点数生成统一插值坐标"""
        return build_wn_ref(self.cut_min, self.cut_max, self.target_points)


DEFAULT_PIPELINE_CONFIG = PipelineConfig()

def get_prefix(name):
    """统一按 letters_sign 规则提取类别前缀，兼容纯字母和字母后缀 +/-"""
    matched = re.match(r"([A-Za-z]+)([+-])?", name)
    if not matched:
        return None
    return f"{matched.group(1)}{matched.group(2) or ''}"

def resolve_pipeline_config(pipeline_config=None):
    """返回离线预处理配置；未传入时使用库内默认配置"""
    return pipeline_config or DEFAULT_PIPELINE_CONFIG

def _resolve_classify_target_dir(root_process_raw, rel_dir, leaf_name):
    """根据叶子目录名推断目标类别目录，统一处理顶层和多级目录"""
    rel_parent = rel_dir.parent
    prefix = get_prefix(leaf_name)
    target_cls = prefix if prefix else leaf_name
    if rel_parent in (Path("."), Path("")):
        return root_process_raw / target_cls
    return root_process_raw / rel_parent / target_cls

def _resolve_group_figure_dir(root_figure, rel_dir):
    """为一个分组解析均值谱图输出目录，避免多处重复拼接父目录"""
    rel_parent = rel_dir.parent
    if rel_parent in (Path("."), Path("")):
        return root_figure
    return root_figure / rel_parent

def _iter_ancestor_level_keys(rel_dir):
    """生成非叶子祖先层级 key，用于高层级均值图聚合"""
    parts = tuple(rel_dir.parts)
    if len(parts) <= 1:
        return
    for level_idx in range(1, len(parts)):
        yield level_idx, parts[:level_idx]

def _safe_plot_name(parts):
    """把层级路径转换成稳定的图片文件名"""
    return "__".join(parts)

def _save_hierarchy_mean_plots(hierarchy_groups, root_figure, cfg):
    """输出 train 阶段聚合得到的高层级平均光谱图"""
    if not hierarchy_groups:
        return 0

    output_root = root_figure / "_hierarchy_mean"
    generated = 0

    for (level_idx, parts), payload in sorted(hierarchy_groups.items()):
        spectra_arr = np.vstack(payload["spectra"])
        level_dir = output_root / f"level_{level_idx}"
        level_dir.mkdir(parents=True, exist_ok=True)

        label = "/".join(parts)
        fig_save_path = level_dir / f"{_safe_plot_name(parts)}.png"
        save_mean_plot(
            wn=payload["wn"],
            spectra=spectra_arr,
            out_path=fig_save_path,
            norm_method=cfg.norm_method,
            bad_bands=COMMON_BAD_BANDS,
            title=f"{label} (mean +/- std, n={spectra_arr.shape[0]})",
        )
        print(f"  Hierarchy mean spectrum saved: {fig_save_path}")
        generated += 1

    return generated

def _save_spectra_files(save_dir, filenames, wn_list, spectra_arr, fmt="%.3f"):
    """批量写出预处理后的光谱文件，统一文本精度和目录创建行为"""
    save_dir.mkdir(parents=True, exist_ok=True)
    for fname, wn_u, sp_u in zip(filenames, wn_list, spectra_arr):
        write_arc_data(save_dir / fname, wn_u, sp_u, fmt=fmt)

def classify(profile, base_dir):
    """将 init 重新归类到 train_raw，统一使用 letters_sign 前缀规则"""
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
    """用 PCA 重构样本并返回逐样本误差，供训练集异常值过滤使用"""
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
    """把 PCA 剔除掉的异常样本写入日志，方便后续追溯"""
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
    """对一个分组内的多条光谱做统一清洗，并按需执行 PCA 异常值过滤"""
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

def build_train(profile, base_dir, pipeline_config=None):
    """从 train_raw 构建 train，并输出每类均值谱图和异常值日志"""
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
    hierarchy_groups = {}

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

        for level_idx, parts in _iter_ancestor_level_keys(rel_dir):
            key = (level_idx, parts)
            if key not in hierarchy_groups:
                hierarchy_groups[key] = {
                    "wn": wn_list[0],
                    "spectra": [],
                }
            hierarchy_groups[key]["spectra"].append(spectra_arr)

    generated_hierarchy_plots = _save_hierarchy_mean_plots(
        hierarchy_groups,
        root_figure,
        cfg,
    )

    print("\nTraining dataset preprocessing finished:")
    print(f"- Clean spectra: {root_process_clean}")
    print(f"- Mean plots: {root_figure}")
    print(f"- Hierarchy mean plots: {generated_hierarchy_plots}")

def preview(profile, base_dir, pipeline_config=None):
    """基于 init 生成预览图，不落盘清洗结果，适合先检查原始数据质量"""
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

def build_test(
    profile,
    base_dir,
    input_dir=None,
    output_dir=None,
    pipeline_config=None,
):
    """从测试原始目录构建 test，并输出每个文件夹的均值谱图"""
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


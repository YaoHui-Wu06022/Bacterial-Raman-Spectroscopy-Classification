import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raman.data.preprocess import preprocess_single_spectrum
from raman.data.io import iter_init_groups, resolve_init_input, write_arc_data
from raman.tool.naming import ensure_name_prefix, extract_letters_prefix
from raman.tool.path import resolve_under_base
from raman.tool.spectrum import build_wn_ref

CUT_MIN = 600
CUT_MAX = 1800
TARGET_POINTS = 896
COMMON_BAD_BANDS = ((890.0, 950.0),)

BASELINE_METHOD = "airPLS"
BASELINE_LAM = 1e5  # 基线平滑强度；越大，估计基线越平滑
BASELINE_ASLS_P = 0.01  # 仅 AsLS 使用的不对称权重；airPLS/arPLS 不使用
BASELINE_MAX_ITER = 15  # 基线迭代次数上限
BASELINE_FIT_MIN = 400  # 基线拟合下限，保留训练范围外缓冲区以稳定边缘基线
BASELINE_FIT_MAX = 2000  # 基线拟合上限，避免更远端异常尖峰污染基线

COSMIC_RAY_ENABLED_PROFILE_IDS = ("shift", "MN_IgA")
COSMIC_RAY_WINDOW_POINTS = 7  # 宇宙射线局部 median/MAD 窗口宽度，单位点
COSMIC_RAY_THRESHOLD = 7.0  # 宇宙射线正残差 z 阈值
COSMIC_RAY_MAX_ITER = 2  # 宇宙射线最大迭代次数

MIN_SAMPLES_PER_CLASS = 8

PCA_ENABLED = True
PCA_COMPONENTS = 0.95
PCA_CENTER = True
PCA_OUTLIER_RATIO = 0.03

@dataclass(frozen=True)
class PipelineConfig:
    """集中管理离线预处理阶段的固定参数，便于 CLI 统一覆盖"""
    cut_min: float = CUT_MIN
    cut_max: float = CUT_MAX
    target_points: int = TARGET_POINTS
    bad_bands: tuple[tuple[float, float], ...] = COMMON_BAD_BANDS
    baseline_method: str = BASELINE_METHOD
    baseline_lam: float = BASELINE_LAM
    baseline_asls_p: float = BASELINE_ASLS_P
    baseline_max_iter: int = BASELINE_MAX_ITER
    baseline_fit_min: float = BASELINE_FIT_MIN
    baseline_fit_max: float = BASELINE_FIT_MAX
    cosmic_ray_enabled_profile_ids: tuple[str, ...] = COSMIC_RAY_ENABLED_PROFILE_IDS
    cosmic_ray_window_points: int = COSMIC_RAY_WINDOW_POINTS
    cosmic_ray_threshold: float = COSMIC_RAY_THRESHOLD
    cosmic_ray_max_iter: int = COSMIC_RAY_MAX_ITER
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS
    pca_enabled: bool = PCA_ENABLED
    pca_components: float | int = PCA_COMPONENTS
    pca_center: bool = PCA_CENTER
    pca_outlier_ratio: float = PCA_OUTLIER_RATIO

    def build_wn_ref(self):
        """根据当前裁剪范围和目标点数生成统一插值坐标"""
        return build_wn_ref(self.cut_min, self.cut_max, self.target_points)


DEFAULT_PIPELINE_CONFIG = PipelineConfig()

def resolve_pipeline_config(pipeline_config=None):
    """返回离线预处理配置"""
    return pipeline_config or DEFAULT_PIPELINE_CONFIG

def _cosmic_ray_enabled(profile, cfg):
    """判断当前数据集是否启用宇宙射线去除"""
    return profile.profile_id in set(cfg.cosmic_ray_enabled_profile_ids)

def _profile_may_use_cosmic_ray(profile, cfg):
    """判断当前 profile 是否可能输出宇宙射线日志"""
    if _cosmic_ray_enabled(profile, cfg):
        return True
    for overrides in (profile.cosmic_ray_overrides or {}).values():
        if bool((overrides or {}).get("enabled", False)):
            return True
    return False

def _cosmic_log_path(profile, base_dir, cfg):
    """只在启用宇宙射线时返回日志路径"""
    if not _profile_may_use_cosmic_ray(profile, cfg):
        return None
    return resolve_under_base(base_dir, profile.cosmic_ray_log_name)

COSMIC_RAY_OVERRIDE_KEY_MAP = {
    "enabled": "cosmic_ray_remove",
    "window_points": "cosmic_ray_window_points",
    "threshold": "cosmic_ray_threshold",
    "max_iter": "cosmic_ray_max_iter",
}

def _matching_cosmic_ray_overrides(profile, label_display=None):
    """按 profile 覆盖表找到当前分组需要叠加的参数"""
    overrides = profile.cosmic_ray_overrides or {}
    label = str(label_display or "").replace("\\", "/").strip("/")
    matched = []
    for scope, values in overrides.items():
        scope_key = str(scope).replace("\\", "/").strip("/")
        if scope_key in {"", "*"}:
            matched.append((0, values))
        elif label == scope_key or label.startswith(f"{scope_key}/"):
            matched.append((scope_key.count("/") + 1, values))
    return [values for _, values in sorted(matched, key=lambda item: item[0])]

def _cast_cosmic_ray_override(current, value):
    """按默认参数类型转换 profile 覆盖值"""
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value

def _apply_cosmic_ray_overrides(options, profile, label_display=None):
    """把 profile 中匹配当前路径的覆盖参数叠加到默认参数上"""
    for overrides in _matching_cosmic_ray_overrides(profile, label_display):
        for key, value in (overrides or {}).items():
            target_key = COSMIC_RAY_OVERRIDE_KEY_MAP.get(key, key)
            if target_key not in options:
                raise KeyError(f"Unknown cosmic ray override key: {key}")
            options[target_key] = _cast_cosmic_ray_override(options[target_key], value)
    return options

def _cosmic_ray_kwargs(profile, cfg, label_display=None):
    """构造单谱宇宙射线去除参数"""
    options = {
        "cosmic_ray_remove": _cosmic_ray_enabled(profile, cfg),
        "cosmic_ray_window_points": int(cfg.cosmic_ray_window_points),
        "cosmic_ray_threshold": float(cfg.cosmic_ray_threshold),
        "cosmic_ray_max_iter": int(cfg.cosmic_ray_max_iter),
    }
    return _apply_cosmic_ray_overrides(options, profile, label_display)


def _resolve_merged_class_dir(root_output, rel_dir, leaf_name):
    """根据叶子小文件夹名推断最终合并类别目录"""
    rel_parent = rel_dir.parent
    prefix = extract_letters_prefix(leaf_name, keep_sign=True)
    target_cls = prefix if prefix else leaf_name
    if rel_parent in (Path("."), Path("")):
        return root_output / target_cls
    return root_output / rel_parent / target_cls

def _save_spectra_files(save_dir, filenames, wn_list, spectra_arr, fmt="%.3f"):
    """批量写出预处理后的光谱文件，统一文本精度和目录创建行为"""
    save_dir.mkdir(parents=True, exist_ok=True)
    for fname, wn_u, sp_u in zip(filenames, wn_list, spectra_arr):
        write_arc_data(save_dir / fname, wn_u, sp_u, fmt=fmt)

def _reset_generated_dir(path):
    """重建生成型目录，避免旧类别残留影响下一次扫描"""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

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

def reset_log_file(log_path):
    """每次进入构建流程时清空旧日志"""
    if log_path is None:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

def _append_log_lines(log_path, lines):
    """把预处理过程中的关键统计追加到指定日志"""
    if log_path is None or not lines:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        for line in lines:
            file.write(f"{line}\n")

def _base_group_stats(input_count, valid_count):
    """构造分组处理统计字段"""
    return {
        "input": input_count,
        "valid_before_pca": valid_count,
        "kept": valid_count,
        "removed": 0,
        "pca_components": 0,
        "threshold": None,
        "skip_reason": None,
        "cosmic_ray_enabled": False,
        "cosmic_single_spectra": 0,
        "cosmic_single_replaced": 0,
    }

def _apply_pca_filter(spectra_arr, filenames, wn_list, stats, label_display, cfg, log_path):
    """对一个已对齐分组执行 PCA 异常样本过滤"""
    if spectra_arr.shape[0] <= 1:
        return spectra_arr, filenames, wn_list

    _, k, errors = pca_reconstruct_and_error(
        spectra_arr,
        n_components=cfg.pca_components,
        center=cfg.pca_center,
    )
    stats["pca_components"] = k
    if k <= 0:
        return spectra_arr, filenames, wn_list

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
    return spectra_arr, filenames, wn_list

def _cosmic_spectra_count(stats):
    return max(int(stats.get("cosmic_single_spectra", 0) or 0), 0)

def _cosmic_avg(stats, key):
    spectra_count = _cosmic_spectra_count(stats)
    if spectra_count <= 0:
        return 0.0
    return float(stats.get(key, 0) or 0) / spectra_count

def _format_cosmic_ray_stats(stats):
    spectra_count = _cosmic_spectra_count(stats)
    return (
        "Cosmic ray replacement avg points/spectrum: "
        f"cosmic_ray={_cosmic_avg(stats, 'cosmic_single_replaced'):.2f}, "
        f"spectra={spectra_count}"
    )

def _print_processing_stats(stats, show_zero_cosmic=False):
    """统一输出 PCA 和宇宙射线清理结果"""
    if stats["pca_components"] > 0:
        print(
            f"  PCA outlier removal: k={stats['pca_components']}, "
            f"threshold={stats['threshold']:.6f}, removed={stats['removed']}"
        )
    if stats.get("cosmic_ray_enabled") and (show_zero_cosmic or stats.get("cosmic_single_replaced", 0) > 0):
        print(f"  {_format_cosmic_ray_stats(stats)}")

def _log_cosmic_ray_stats(label_display, stats, log_path, show_zero_cosmic=False):
    """把每个小文件夹的宇宙射线清理统计写入日志"""
    if not stats.get("cosmic_ray_enabled"):
        return
    if not show_zero_cosmic and stats.get("cosmic_single_replaced", 0) <= 0:
        return
    lines = [f"[{label_display}] {_format_cosmic_ray_stats(stats)}"]
    _append_log_lines(log_path, lines)

def _finalize_group_result(
    spectra,
    wn_list,
    filenames,
    stats,
    min_samples,
    apply_pca,
    label_display,
    cfg,
    log_path,
):
    """统一完成分组数量检查、PCA 过滤和返回结构组装"""
    if len(spectra) < min_samples:
        stats["skip_reason"] = "too_few_preprocessed"
        return None, stats

    spectra_arr = np.vstack(spectra)
    if apply_pca:
        spectra_arr, filenames, wn_list = _apply_pca_filter(
            spectra_arr,
            filenames,
            wn_list,
            stats,
            label_display,
            cfg,
            log_path,
        )

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

def preprocess_physical_group(profile, cfg, samples, label_display, min_samples=1):
    """执行小文件夹内物理清洗，不做 PCA"""
    wn_ref = cfg.build_wn_ref()
    spectra, wn_list, filenames = [], [], []
    cosmic_ray_options = _cosmic_ray_kwargs(profile, cfg, label_display)
    cosmic_ray_enabled = bool(cosmic_ray_options.get("cosmic_ray_remove"))
    cosmic_single_spectra = 0
    cosmic_single_replaced = 0

    for fname, wn, sp in samples:
        if wn.size == 0 or sp.size == 0:
            continue

        if cosmic_ray_enabled:
            cosmic_single_spectra += 1
        wn_u, sp_u, single_replaced = preprocess_single_spectrum(
            wn,
            sp,
            cut_min=cfg.cut_min,
            cut_max=cfg.cut_max,
            wn_ref=wn_ref,
            bad_bands=cfg.bad_bands,
            baseline_method=cfg.baseline_method,
            baseline_lam=cfg.baseline_lam,
            baseline_asls_p=cfg.baseline_asls_p,
            baseline_max_iter=cfg.baseline_max_iter,
            baseline_fit_min=cfg.baseline_fit_min,
            baseline_fit_max=cfg.baseline_fit_max,
            **cosmic_ray_options,
        )
        if cosmic_ray_enabled:
            cosmic_single_replaced += int(single_replaced)
        if wn_u is None:
            continue

        spectra.append(sp_u)
        wn_list.append(wn_u)
        filenames.append(fname)

    stats = _base_group_stats(len(samples), len(spectra))
    stats["cosmic_ray_enabled"] = cosmic_ray_enabled
    stats["cosmic_single_spectra"] = int(cosmic_single_spectra)
    stats["cosmic_single_replaced"] = int(cosmic_single_replaced)

    return _finalize_group_result(
        spectra,
        wn_list,
        filenames,
        stats,
        min_samples,
        False,
        label_display,
        cfg,
        None,
    )

def finalize_clean_group_samples(
    samples,
    label_display,
    min_samples=None,
    log_path=None,
    apply_pca=None,
    pipeline_config=None,
):
    """对已经物理清洗好的光谱分组执行可选 PCA 过滤"""
    cfg = resolve_pipeline_config(pipeline_config)
    if min_samples is None:
        min_samples = int(cfg.min_samples_per_class)
    if apply_pca is None:
        apply_pca = bool(cfg.pca_enabled)

    spectra, wn_list, filenames = [], [], []
    for fname, wn, sp in samples:
        if wn.size == 0 or sp.size == 0:
            continue
        spectra.append(np.asarray(sp, dtype=np.float32))
        wn_list.append(wn)
        filenames.append(fname)

    stats = _base_group_stats(len(samples), len(spectra))

    return _finalize_group_result(
        spectra,
        wn_list,
        filenames,
        stats,
        min_samples,
        apply_pca,
        label_display,
        cfg,
        log_path,
    )

def _collect_merged_init_groups(profile, input_path, root_process_clean, cfg, cosmic_log_path):
    """从 init 直接物理清洗并按叶子名前缀合并为最终类别"""
    groups = {}
    skipped = 0
    for rel_dir, leaf_name, samples in iter_init_groups(input_path):
        label = rel_dir.as_posix() if rel_dir != Path(".") else leaf_name
        label_display = label.replace("\\", "/")
        print(f"\n=== Build train source: {label_display} ===")

        processed_group, stats = preprocess_physical_group(
            profile,
            cfg,
            samples,
            label_display,
        )

        if stats["skip_reason"] is not None:
            print(
                f"  Skip: no valid spectra after preprocessing "
                f"({stats['valid_before_pca']}/{stats['input']})"
            )
            skipped += 1
            continue

        _print_processing_stats(stats)
        _log_cosmic_ray_stats(label_display, stats, cosmic_log_path)

        target_dir = _resolve_merged_class_dir(
            root_process_clean,
            rel_dir,
            leaf_name,
        )
        target_rel_dir = target_dir.relative_to(root_process_clean)
        group = groups.setdefault(
            target_rel_dir.as_posix(),
            {
                "target_dir": target_dir,
                "rel_dir": target_rel_dir,
                "samples": [],
            },
        )

        for fname, wn, sp in zip(
            processed_group["filenames"],
            processed_group["wn_list"],
            processed_group["spectra"],
        ):
            out_name = ensure_name_prefix(leaf_name, fname)
            group["samples"].append((out_name, wn, sp))

    return groups, skipped

def build_train(profile, base_dir, pipeline_config=None):
    """从 init 直接清洗、按类别合并后执行 PCA 并生成最终 train"""
    cfg = resolve_pipeline_config(pipeline_config)
    base_dir = Path(base_dir)
    input_path = resolve_init_input(base_dir, profile)
    root_process_clean = resolve_under_base(base_dir, profile.root_train_clean)
    root_figure = resolve_under_base(base_dir, profile.root_train_fig)
    pca_log_path = resolve_under_base(base_dir, profile.pca_log_name)
    cosmic_log_path = _cosmic_log_path(profile, base_dir, cfg)
    reset_log_file(pca_log_path)

    _reset_generated_dir(root_process_clean)
    merged_groups, skipped_sources = _collect_merged_init_groups(
        profile,
        input_path,
        root_process_clean,
        cfg,
        cosmic_log_path,
    )

    for group in sorted(merged_groups.values(), key=lambda item: item["rel_dir"].as_posix()):
        rel_dir = group["rel_dir"]
        label_display = rel_dir.as_posix()
        print(f"\n=== Build train: {label_display} ===")

        processed_group, stats = finalize_clean_group_samples(
            samples=group["samples"],
            label_display=label_display,
            log_path=pca_log_path,
            pipeline_config=cfg,
        )

        if stats["skip_reason"] == "too_few_preprocessed":
            print(
                f"  Skip: too few samples ({stats['valid_before_pca']}) in {label_display}"
            )
            continue

        _print_processing_stats(stats)

        if stats["skip_reason"] == "too_few_after_pca":
            print(f"  Skip: too few samples ({stats['kept']}) in {label_display}")
            continue

        spectra_arr = processed_group["spectra"]
        filenames = processed_group["filenames"]
        wn_list = processed_group["wn_list"]

        save_dir = group["target_dir"]
        _save_spectra_files(save_dir, filenames, wn_list, spectra_arr)

    _reset_generated_dir(root_figure)

    print("\nTraining dataset preprocessing finished:")
    print(f"- Final train spectra: {root_process_clean}")
    print(f"- Cleared stale train plots: {root_figure}")
    print(f"- PCA log: {pca_log_path}")
    print(f"- Skipped source groups: {skipped_sources}")
    if cosmic_log_path is not None and cosmic_log_path.is_file() and cosmic_log_path.stat().st_size > 0:
        print(f"- Cosmic ray log: {cosmic_log_path}")


def build_test(profile, base_dir, pipeline_config=None):
    """从 init_test 生成已预处理的独立测试集 test，不做 PCA"""
    cfg = resolve_pipeline_config(pipeline_config)
    base_dir = Path(base_dir)
    root_init_test = resolve_under_base(base_dir, profile.root_init_test)
    root_test = resolve_under_base(base_dir, profile.root_test)
    cosmic_log_path = _cosmic_log_path(profile, base_dir, cfg)

    if not root_init_test.is_dir():
        raise FileNotFoundError(f"Missing init_test folder: {root_init_test}")

    _reset_generated_dir(root_test)
    built_groups = 0
    skipped_groups = 0
    total_files = 0

    for rel_dir, leaf_name, samples in iter_init_groups(root_init_test):
        label = rel_dir.as_posix() if rel_dir != Path(".") else leaf_name
        label_display = label.replace("\\", "/")
        print(f"\n=== Build test: {label_display} ===")

        processed_group, stats = preprocess_physical_group(
            profile,
            cfg,
            samples,
            label_display,
            min_samples=1,
        )
        if stats["skip_reason"] is not None:
            print(
                f"  Skip: no valid spectra after preprocessing "
                f"({stats['valid_before_pca']}/{stats['input']})"
            )
            skipped_groups += 1
            continue

        _print_processing_stats(stats)
        _log_cosmic_ray_stats(label_display, stats, cosmic_log_path)

        save_dir = root_test / rel_dir
        spectra_arr = processed_group["spectra"]
        filenames = processed_group["filenames"]
        wn_list = processed_group["wn_list"]
        _save_spectra_files(save_dir, filenames, wn_list, spectra_arr)
        built_groups += 1
        total_files += len(filenames)

    print("\nTest dataset preprocessing finished:")
    print(f"- Source init_test: {root_init_test}")
    print(f"- Final test spectra: {root_test}")
    print(f"- Groups built: {built_groups}")
    print(f"- Spectra built: {total_files}")
    print(f"- Skipped groups: {skipped_groups}")
    if cosmic_log_path is not None and cosmic_log_path.is_file() and cosmic_log_path.stat().st_size > 0:
        print(f"- Cosmic ray log: {cosmic_log_path}")

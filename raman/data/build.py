import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from raman.data.archive import (
    iter_arc_dirs,
    iter_init_groups,
    resolve_init_input,
    resolve_path,
)
from raman.data.offline import (
    preprocess_single_spectrum,
    remove_group_cosmic_rays,
    save_mean_plot,
)
from raman.data.spectrum import (
    build_wn_ref,
    read_arc_data,
    write_arc_data,
)

CUT_MIN = 600
CUT_MAX = 1800
TARGET_POINTS = 896
COMMON_BAD_BANDS = ((890.0, 950.0),)

ASLS_LAM = 3e5 # 更改
ASLS_P = 0.005 # 更改
ASLS_MAX_ITER = 15

COSMIC_RAY_ENABLED_PROFILE_IDS = ("bacteria", "delete", "Enterobacteriaceae")
COSMIC_RAY_WINDOW = 7  # 局部窗口
COSMIC_RAY_THRESHOLD = 8.0  #异常阈值
COSMIC_RAY_MAX_ITER = 2  # 最大迭代次数
COSMIC_RAY_GROUP_THRESHOLD = 15  # 组内兜底异常阈值
COSMIC_RAY_GROUP_MIN_SAMPLES = 10  # 少于该数量时不做组内统计

MIN_SAMPLES_PER_CLASS = 8
PLOT_NORM_METHOD = "snv"

PCA_ENABLED = True
PCA_COMPONENTS = 0.95
PCA_CENTER = True
PCA_OUTLIER_RATIO = 0.03

TRAIN_RAW_CONFIG_NAME = "config.json"

@dataclass(frozen=True)
class PipelineConfig:
    """集中管理离线预处理阶段的固定参数，便于 CLI 统一覆盖"""
    cut_min: float = CUT_MIN
    cut_max: float = CUT_MAX
    target_points: int = TARGET_POINTS
    bad_bands: tuple[tuple[float, float], ...] = COMMON_BAD_BANDS
    asls_lam: float = ASLS_LAM
    asls_p: float = ASLS_P
    asls_max_iter: int = ASLS_MAX_ITER
    cosmic_ray_enabled_profile_ids: tuple[str, ...] = COSMIC_RAY_ENABLED_PROFILE_IDS
    cosmic_ray_window: int = COSMIC_RAY_WINDOW
    cosmic_ray_threshold: float = COSMIC_RAY_THRESHOLD
    cosmic_ray_max_iter: int = COSMIC_RAY_MAX_ITER
    cosmic_ray_group_threshold: float = COSMIC_RAY_GROUP_THRESHOLD
    cosmic_ray_group_min_samples: int = COSMIC_RAY_GROUP_MIN_SAMPLES
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS
    plot_norm_method: str = PLOT_NORM_METHOD
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

def _json_ready(value):
    """把 dataclass/tuple 等配置值转换成稳定 JSON 结构"""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))

def _train_raw_config_payload(profile, cfg):
    """生成 train_raw 中间层的参数指纹"""
    return {
        "profile_id": profile.profile_id,
        "pipeline_config": _json_ready(asdict(cfg)),
    }

def _train_raw_config_path(root_process_raw):
    return Path(root_process_raw) / TRAIN_RAW_CONFIG_NAME

def _write_train_raw_config(root_process_raw, profile, cfg):
    """在 train_raw 中记录生成该中间层的配置"""
    config_path = _train_raw_config_path(root_process_raw)
    config_path.write_text(
        json.dumps(
            _train_raw_config_payload(profile, cfg),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

def _train_raw_config_matches(root_process_raw, profile, cfg):
    """判断现有 train_raw 是否由当前配置生成"""
    config_path = _train_raw_config_path(root_process_raw)
    if not config_path.is_file():
        return False
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == _train_raw_config_payload(profile, cfg)

def _cosmic_ray_enabled(profile, cfg):
    """判断当前数据集是否启用宇宙射线去除"""
    return profile.profile_id in set(cfg.cosmic_ray_enabled_profile_ids)

def _cosmic_ray_kwargs(profile, cfg):
    """构造单谱宇宙射线去除参数"""
    return {
        "cosmic_ray_remove": _cosmic_ray_enabled(profile, cfg),
        "cosmic_ray_window": int(cfg.cosmic_ray_window),
        "cosmic_ray_threshold": float(cfg.cosmic_ray_threshold),
        "cosmic_ray_max_iter": int(cfg.cosmic_ray_max_iter),
    }


def _cosmic_ray_group_kwargs(profile, cfg):
    """构造小文件夹内宇宙射线兜底参数"""
    return {
        "enabled": _cosmic_ray_enabled(profile, cfg),
        "threshold": float(cfg.cosmic_ray_group_threshold),
        "min_samples": int(cfg.cosmic_ray_group_min_samples),
    }


def _resolve_merged_class_dir(root_output, rel_dir, leaf_name):
    """根据叶子小文件夹名推断最终合并类别目录"""
    rel_parent = rel_dir.parent
    prefix = get_prefix(leaf_name)
    target_cls = prefix if prefix else leaf_name
    if rel_parent in (Path("."), Path("")):
        return root_output / target_cls
    return root_output / rel_parent / target_cls

def _with_leaf_prefix(leaf_name, filename):
    """合并多个小文件夹时避免文件名冲突"""
    prefix = f"{leaf_name}_"
    return filename if filename.startswith(prefix) else f"{prefix}{filename}"

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
            norm_method=cfg.plot_norm_method,
            bad_bands=cfg.bad_bands,
            title=f"{label} (mean, q10-q90, n={spectra_arr.shape[0]})",
        )
        print(f"  Hierarchy mean spectrum saved: {fig_save_path}")
        generated += 1

    return generated

def _save_spectra_files(save_dir, filenames, wn_list, spectra_arr, fmt="%.3f"):
    """批量写出预处理后的光谱文件，统一文本精度和目录创建行为"""
    save_dir.mkdir(parents=True, exist_ok=True)
    for fname, wn_u, sp_u in zip(filenames, wn_list, spectra_arr):
        write_arc_data(save_dir / fname, wn_u, sp_u, fmt=fmt)

def _save_mean_figure(root_figure, rel_dir, filename, wn, spectra, title, cfg):
    """统一保存均值谱图，SNV 只在这里用于展示"""
    fig_dir = _resolve_group_figure_dir(root_figure, rel_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / filename
    save_mean_plot(
        wn=wn,
        spectra=spectra,
        out_path=fig_path,
        norm_method=cfg.plot_norm_method,
        bad_bands=cfg.bad_bands,
        title=title,
    )
    print(f"  Mean spectrum saved: {fig_path}")
    return fig_path

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
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

def _append_log_lines(log_path, lines):
    """把预处理过程中的关键统计追加到 log.txt"""
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
        "cosmic_single_replaced": 0,
        "cosmic_group_replaced": 0,
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

def _print_processing_stats(stats, show_zero_cosmic=False):
    """统一输出 PCA 和宇宙射线清理结果"""
    if stats["pca_components"] > 0:
        print(
            f"  PCA outlier removal: k={stats['pca_components']}, "
            f"threshold={stats['threshold']:.6f}, removed={stats['removed']}"
        )
    if show_zero_cosmic or stats.get("cosmic_single_replaced", 0) > 0:
        print(
            "  Cosmic ray single cleanup: "
            f"replaced={stats['cosmic_single_replaced']}"
        )
    if show_zero_cosmic or stats.get("cosmic_group_replaced", 0) > 0:
        print(
            "  Cosmic ray group cleanup: "
            f"replaced={stats['cosmic_group_replaced']}"
        )

def _log_cosmic_ray_stats(label_display, stats, log_path, show_zero_cosmic=False):
    """把每个小文件夹的宇宙射线清理统计写入日志"""
    lines = []
    if show_zero_cosmic or stats.get("cosmic_single_replaced", 0) > 0:
        lines.append(
            f"[{label_display}] Cosmic ray single cleanup: "
            f"replaced={stats['cosmic_single_replaced']}"
        )
    if show_zero_cosmic or stats.get("cosmic_group_replaced", 0) > 0:
        lines.append(
            f"[{label_display}] Cosmic ray group cleanup: "
            f"replaced={stats['cosmic_group_replaced']}"
        )
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
    cosmic_ray_options = _cosmic_ray_kwargs(profile, cfg)
    cosmic_ray_group_options = _cosmic_ray_group_kwargs(profile, cfg)
    cosmic_single_replaced = 0

    for fname, wn, sp in samples:
        if wn.size == 0 or sp.size == 0:
            continue

        wn_u, sp_u, single_replaced = preprocess_single_spectrum(
            wn,
            sp,
            cut_min=cfg.cut_min,
            cut_max=cfg.cut_max,
            wn_ref=wn_ref,
            bad_bands=cfg.bad_bands,
            asls_lam=cfg.asls_lam,
            asls_p=cfg.asls_p,
            asls_max_iter=cfg.asls_max_iter,
            **cosmic_ray_options,
        )
        cosmic_single_replaced += int(single_replaced)
        if wn_u is None:
            continue

        spectra.append(sp_u)
        wn_list.append(wn_u)
        filenames.append(fname)

    stats = _base_group_stats(len(samples), len(spectra))
    stats["cosmic_single_replaced"] = int(cosmic_single_replaced)

    if len(spectra) >= min_samples and cosmic_ray_group_options.get("enabled", False):
        spectra_arr = np.vstack(spectra)
        spectra_arr, replaced = remove_group_cosmic_rays(
            spectra_arr,
            threshold=cosmic_ray_group_options.get(
                "threshold",
                cfg.cosmic_ray_group_threshold,
            ),
            min_samples=cosmic_ray_group_options.get(
                "min_samples",
                cfg.cosmic_ray_group_min_samples,
            ),
        )
        spectra = list(spectra_arr)
        stats["cosmic_group_replaced"] = int(replaced)

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

def _has_arc_data(root_dir):
    """递归判断目录中是否已有可复用光谱文件"""
    root_dir = Path(root_dir)
    return root_dir.exists() and any(root_dir.rglob("*.arc_data"))

def build_train_raw(profile, base_dir, pipeline_config=None, log_path=None):
    """从 init 生成按小文件夹保存的物理清洗中间层 train_raw"""
    cfg = resolve_pipeline_config(pipeline_config)
    base_dir = Path(base_dir)
    input_path = resolve_init_input(base_dir, profile)
    root_process_raw = resolve_path(base_dir, profile.root_process_raw)
    _reset_generated_dir(root_process_raw)

    generated = 0
    skipped = 0

    for rel_dir, leaf_name, samples in iter_init_groups(input_path):
        label = rel_dir.as_posix() if rel_dir != Path(".") else leaf_name
        label_display = label.replace("\\", "/")
        print(f"\n=== Build train_raw: {label_display} ===")

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
        _log_cosmic_ray_stats(label_display, stats, log_path)

        save_dir = root_process_raw / rel_dir
        _save_spectra_files(
            save_dir,
            processed_group["filenames"],
            processed_group["wn_list"],
            processed_group["spectra"],
        )
        generated += len(processed_group["filenames"])

    print("\nTrain raw preprocessing finished:")
    print(f"- Clean intermediate spectra: {root_process_raw}")
    print(f"- Generated={generated}, Skipped groups={skipped}")
    _write_train_raw_config(root_process_raw, profile, cfg)
    print(f"- Config: {_train_raw_config_path(root_process_raw)}")

def _collect_merged_train_groups(root_process_raw, root_process_clean):
    """从 train_raw 小文件夹收集样本，并按叶子名前缀合并为最终类别"""
    groups = {}
    for leaf_dir, arc_files in iter_arc_dirs(root_process_raw):
        rel_dir = leaf_dir.relative_to(root_process_raw)
        leaf_name = leaf_dir.name
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

        for fname in arc_files:
            wn, sp = read_arc_data(leaf_dir / fname)
            out_name = _with_leaf_prefix(leaf_name, fname)
            group["samples"].append((out_name, wn, sp))

    return groups

def _read_arc_samples(root, arc_files, input_root):
    """读取一个小文件夹内的光谱，并统计读取失败数量"""
    samples = []
    errored = 0
    for fname in arc_files:
        in_path = root / fname
        try:
            wn, sp = read_arc_data(in_path)
            samples.append((fname, wn, sp))
        except Exception as exc:
            rel_path = in_path.relative_to(input_root).as_posix()
            print(f"[ERROR] {rel_path}: {exc}")
            errored += 1
    return samples, errored

def build_train(profile, base_dir, pipeline_config=None):
    """复用 train_raw，按类别合并后执行 PCA 并生成最终 train"""
    cfg = resolve_pipeline_config(pipeline_config)
    base_dir = Path(base_dir)
    root_process_raw = resolve_path(base_dir, profile.root_process_raw)
    root_process_clean = resolve_path(base_dir, profile.root_train_clean)
    root_figure = resolve_path(base_dir, profile.root_train_fig)
    log_path = resolve_path(base_dir, profile.log_name)
    reset_log_file(log_path)

    if not _has_arc_data(root_process_raw):
        print(f"No reusable train_raw found, build from init: {root_process_raw}")
        build_train_raw(profile, base_dir, pipeline_config=cfg, log_path=log_path)
    elif not _train_raw_config_matches(root_process_raw, profile, cfg):
        print(f"train_raw config changed, rebuild from init: {root_process_raw}")
        build_train_raw(profile, base_dir, pipeline_config=cfg, log_path=log_path)
    else:
        print(f"Reuse existing train_raw: {root_process_raw}")

    if not _has_arc_data(root_process_raw):
        raise FileNotFoundError(f"No .arc_data files found in: {root_process_raw}")

    _reset_generated_dir(root_process_clean)
    _reset_generated_dir(root_figure)
    hierarchy_groups = {}
    merged_groups = _collect_merged_train_groups(root_process_raw, root_process_clean)

    for group in sorted(merged_groups.values(), key=lambda item: item["rel_dir"].as_posix()):
        rel_dir = group["rel_dir"]
        label_display = rel_dir.as_posix()
        print(f"\n=== Build train: {label_display} ===")

        processed_group, stats = finalize_clean_group_samples(
            samples=group["samples"],
            label_display=label_display,
            log_path=log_path,
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

        title = " - ".join(rel_dir.parts) + " (mean, q10-q90)"
        _save_mean_figure(
            root_figure=root_figure,
            rel_dir=rel_dir,
            filename=f"{rel_dir.name}.png",
            wn=wn_list[0],
            spectra=spectra_arr,
            title=title,
            cfg=cfg,
        )

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
    print(f"- Reusable clean intermediate spectra: {root_process_raw}")
    print(f"- Final train spectra: {root_process_clean}")
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

        _print_processing_stats(stats, show_zero_cosmic=True)

        title = " - ".join(rel_dir.parts) if rel_dir != Path(".") else leaf_name
        title = (
            f"{title} (mean, q10-q90, kept {stats['kept']}/{stats['input']})"
        )

        _save_mean_figure(
            root_figure=root_init_fig,
            rel_dir=rel_dir,
            filename=f"{leaf_name}.png",
            wn=processed_group["wn"],
            spectra=processed_group["spectra"],
            title=title,
            cfg=cfg,
        )
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

    _reset_generated_dir(output_dir)
    _reset_generated_dir(root_test_fig)

    processed = 0
    skipped = 0
    errored = 0

    for root, arc_files in iter_arc_dirs(input_dir):
        rel_dir = root.relative_to(input_dir)
        label_display = rel_dir.as_posix()
        samples, group_errors = _read_arc_samples(root, arc_files, input_dir)
        errored += group_errors

        processed_group, stats = preprocess_physical_group(
            profile,
            cfg,
            samples,
            label_display,
        )
        skipped += stats["input"] - stats["valid_before_pca"]

        if stats["skip_reason"] is not None:
            print(
                f"  Skip: no valid spectra after preprocessing "
                f"({stats['valid_before_pca']}/{stats['input']})"
            )
            continue

        _print_processing_stats(stats)
        _save_spectra_files(
            output_dir / rel_dir,
            processed_group["filenames"],
            processed_group["wn_list"],
            processed_group["spectra"],
        )
        processed += len(processed_group["filenames"])

        _save_mean_figure(
            root_figure=root_test_fig,
            rel_dir=rel_dir,
            filename=f"{root.name}.png",
            wn=processed_group["wn"],
            spectra=processed_group["spectra"],
            title=f"{label_display} (mean, q10-q90)",
            cfg=cfg,
        )

    print(
        "Test dataset preprocessing finished. "
        f"Processed={processed}, Skipped={skipped}, Error={errored}"
    )

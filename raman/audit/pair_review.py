"""绘制指定文件夹中 000/001 重复测量的一致性复核图。"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from raman.audit.common import preprocess_spectrum_for_audit
from raman.pipeline import DEFAULT_PIPELINE_CONFIG
from raman.tool.dataset import resolve_dataset
from raman.tool.path import PROJECT_ROOT
from raman.tool.plotting import add_bad_band_spans, plot_segments_without_bad_bands


DEFAULT_FOLDERS = ("EAB02", "EAB03", "ECL03", "SMA03")
PAIR_PATTERN = re.compile(r"^(.*)_(000|001)_shift_cos\.arc_data$", re.IGNORECASE)
SOURCE_ROOTS = ("init", "delete/stage2", "delete/stage3")


def _pair_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """计算两条处理谱的 Pearson 相关性。"""
    first = first - first.mean()
    second = second - second.mean()
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-8)
    return float(first @ second / denominator)


def _target_directories(dataset_dir: Path, folders: tuple[str, ...]):
    """在保留区和自动删除区查找指定的小文件夹。"""
    for source_root in SOURCE_ROOTS:
        root = dataset_dir / source_root
        if not root.is_dir():
            continue
        for genus_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for folder in folders:
                target = genus_dir / folder
                if target.is_dir():
                    yield source_root, genus_dir.name, target


def _load_pairs(dataset_dir: Path, folders: tuple[str, ...]):
    """读取可组成 000/001 对的处理谱，并保留它们当前所在位置。"""
    profile, profile_dir = resolve_dataset("alldata", PROJECT_ROOT)
    if profile_dir != dataset_dir:
        raise ValueError(f"数据集目录不匹配：{dataset_dir}")

    grouped: dict[tuple[str, str, str], dict[str, tuple[Path, str, np.ndarray]]] = defaultdict(dict)
    for source_root, genus, folder_dir in _target_directories(dataset_dir, folders):
        for path in sorted(folder_dir.glob("*.arc_data")):
            match = PAIR_PATTERN.match(path.name)
            if match is None:
                continue
            payload = preprocess_spectrum_for_audit(path, profile, DEFAULT_PIPELINE_CONFIG)
            if payload.get("skip_reason"):
                continue
            key = (genus, folder_dir.name, match.group(1))
            side = match.group(2)
            if side in grouped[key]:
                raise ValueError(f"重复的 {side} 测量：{path}")
            grouped[key][side] = (path, source_root, np.asarray(payload["z"], dtype=np.float32))

    pairs = []
    for (genus, folder, cell), sides in sorted(grouped.items()):
        if set(sides) != {"000", "001"}:
            continue
        first_path, first_source, first = sides["000"]
        second_path, second_source, second = sides["001"]
        pairs.append(
            {
                "genus": genus,
                "folder": folder,
                "cell": cell,
                "first_path": first_path,
                "second_path": second_path,
                "first_source": first_source,
                "second_source": second_source,
                "first": first,
                "second": second,
                "corr": _pair_correlation(first, second),
            }
        )
    return pairs


def _write_scores(path: Path, pairs: list[dict[str, object]], selected: set[tuple[str, str, str]]) -> None:
    """写出全部成对一致性分数与本次绘图选择。"""
    rows = [
        {
            "genus": item["genus"],
            "folder": item["folder"],
            "cell": item["cell"],
            "pair_corr": f"{item['corr']:.6f}",
            "first_rel_path": Path(item["first_path"]).as_posix(),
            "second_rel_path": Path(item["second_path"]).as_posix(),
            "first_source": item["first_source"],
            "second_source": item["second_source"],
            "drawn": (item["genus"], item["folder"], item["cell"]) in selected,
        }
        for item in sorted(pairs, key=lambda value: (value["folder"], value["corr"], value["cell"]))
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _plot_pair(item: dict[str, object], output: Path, rank: int) -> None:
    """绘制一对重复测量的归一化谱与残差。"""
    grid = DEFAULT_PIPELINE_CONFIG.build_wn_ref()
    bad_bands = DEFAULT_PIPELINE_CONFIG.bad_bands
    for low, high in bad_bands:
        grid = grid[(grid < low) | (grid > high)]
    first = np.asarray(item["first"])
    second = np.asarray(item["second"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    plot_segments_without_bad_bands(axes[0], grid, first, bad_bands, show_bad_bands=False, color="C0", lw=0.9, label="000")
    plot_segments_without_bad_bands(axes[0], grid, second, bad_bands, show_bad_bands=False, color="C3", lw=0.9, label="001")
    add_bad_band_spans(axes[0], bad_bands, alpha=0.35)
    axes[0].legend()
    residual = first - second
    plot_segments_without_bad_bands(axes[1], grid, residual, bad_bands, show_bad_bands=False, color="C4", lw=0.8)
    add_bad_band_spans(axes[1], bad_bands, alpha=0.35)
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_title(
        f"{item['genus']}/{item['folder']} | {item['cell']} | "
        f"pair corr={item['corr']:.4f} | low-rank={rank}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_pair_review(folders: tuple[str, ...] = DEFAULT_FOLDERS, max_per_folder: int = 6) -> Path:
    """输出指定文件夹中相关性最低的重复测量对图。"""
    if max_per_folder < 1:
        raise ValueError("每个文件夹至少绘制一对重复测量")
    _, dataset_dir = resolve_dataset("alldata", PROJECT_ROOT)
    pairs = _load_pairs(dataset_dir, folders)
    by_folder: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in pairs:
        by_folder[(str(item["genus"]), str(item["folder"]))].append(item)
    selected_pairs = []
    for items in by_folder.values():
        selected_pairs.extend(sorted(items, key=lambda value: value["corr"])[:max_per_folder])
    selected = {(str(item["genus"]), str(item["folder"]), str(item["cell"])) for item in selected_pairs}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = dataset_dir / "audit_runs" / f"{stamp}_pair_review"
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_scores(out_dir / "pair_scores.csv", pairs, selected)
    ranks: dict[tuple[str, str], int] = defaultdict(int)
    for item in sorted(selected_pairs, key=lambda value: (value["folder"], value["corr"], value["cell"])):
        key = (str(item["genus"]), str(item["folder"]))
        ranks[key] += 1
        output = out_dir / "figures" / key[0] / key[1] / f"{ranks[key]:02d}_{item['cell']}.png"
        _plot_pair(item, output, ranks[key])
    return out_dir


def main(argv=None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="绘制低一致性 000/001 重复测量对")
    parser.add_argument("--max-per-folder", type=int, default=6, help="每个文件夹绘制相关性最低的对数")
    args = parser.parse_args(argv)
    print(run_pair_review(max_per_folder=args.max_per_folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

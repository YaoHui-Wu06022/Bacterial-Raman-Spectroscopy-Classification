"""同属同前缀参考组审核入口"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raman.audit.common import (
    PROJECT_ROOT,
    plot_segments_without_bad_bands,
    preprocess_spectrum_for_audit,
    prefix_of,
    resolve_audit_folder,
    resolve_dataset,
)
from raman.audit.config import DEFAULT_AUDIT_CONFIG
from raman.audit.scoring import score_reference_rows
from raman.data.build import DEFAULT_PIPELINE_CONFIG


class Processor:
    """带缓存的单谱预处理器，避免参考谱重复处理"""

    def __init__(self, profile, cfg):
        self.profile = profile
        self.cfg = cfg
        self.wn_ref = cfg.build_wn_ref()
        self.cache: dict[Path, tuple[np.ndarray, object] | None] = {}

    def load(self, path: Path) -> tuple[np.ndarray, object] | None:
        path = path.resolve()
        if path in self.cache:
            return self.cache[path]
        payload = preprocess_spectrum_for_audit(path, self.profile, self.cfg, wn_ref=self.wn_ref)
        if payload["skip_reason"]:
            self.cache[path] = None
            return None
        self.cache[path] = (np.asarray(payload["z"], dtype=np.float32), payload["cosmic_stats"])
        return self.cache[path]


def reference_files(dataset_root: Path, target_folder: Path, audit_cfg=DEFAULT_AUDIT_CONFIG):
    """选择参考谱，优先同属同前缀，数量不足时退回同属"""
    rel = target_folder.relative_to(dataset_root)
    genus = rel.parts[0]
    folder_name = rel.parts[1]
    genus_root = dataset_root / genus
    target_prefix = prefix_of(folder_name)

    folders = [path for path in sorted(genus_root.iterdir()) if path.is_dir() and path != target_folder]
    same_prefix = [path for path in folders if prefix_of(path.name) == target_prefix]
    chosen = same_prefix if sum(len(list(path.glob("*.arc_data"))) for path in same_prefix) >= audit_cfg.min_ref_files else folders

    files: list[Path] = []
    for folder in chosen:
        files.extend(sorted(folder.glob("*.arc_data")))
    return files, [folder.name for folder in chosen]


def _write_outputs(out_root, target_folder, dataset_root, ref_dirs, rows, thresholds, no_plot, target_arr, ref_median, folder_median, cfg):
    rel = target_folder.relative_to(dataset_root)
    out_dir = out_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scores.csv"
    json_path = out_dir / "summary.json"
    png_path = out_dir / "review.png"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "folder": rel.as_posix(),
        "target_count": len(rows),
        "reference_dirs": ref_dirs,
        "thresholds": thresholds,
        "remove_count": sum(row["decision"] == "remove" for row in rows),
        "keep_count": sum(row["decision"] == "keep" for row in rows),
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    if no_plot:
        return csv_path, json_path, None

    wn = cfg.build_wn_ref()
    keep = np.ones_like(wn, dtype=bool)
    for band_min, band_max in cfg.bad_bands:
        keep &= ~((wn >= band_min) & (wn <= band_max))
    wn = wn[keep]

    plot_items = sorted(zip(rows, target_arr), key=lambda item: (item[0]["decision"] != "remove", item[0]["corr_ref"]))
    cols = 3
    nrows = int(np.ceil(len(plot_items) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(15, max(3, nrows * 2.4)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (row, spec) in zip(axes, plot_items):
        color = "#D62728" if row["decision"] == "remove" else "#1F77B4"
        plot_segments_without_bad_bands(ax, wn, ref_median, cfg.bad_bands, color="#444444", lw=1.1, label="ref median")
        plot_segments_without_bad_bands(ax, wn, folder_median, cfg.bad_bands, color="#0C7C59", lw=1.0, label="folder median")
        plot_segments_without_bad_bands(ax, wn, spec, cfg.bad_bands, color=color, lw=0.9)
        ax.set_title(
            f"{row['file']} | {row['decision']}\n"
            f"r={row['corr_ref']:.2f}, rf={row['corr_folder']:.2f}, "
            f"bad={row['bad_ratio_z6']:.2f}, cos={row['cosmic_total']}",
            fontsize=8,
        )
        ax.grid(alpha=0.2)
    for ax in axes[len(plot_items) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{rel.as_posix()} single-spectrum review", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, json_path, png_path


def audit_folder(dataset, folder, dataset_root=None, out_dir=None, no_plot=False, audit_config=None):
    """执行参考组离群审核"""
    audit_cfg = audit_config or DEFAULT_AUDIT_CONFIG
    profile, dataset_dir = resolve_dataset(dataset, PROJECT_ROOT)
    cfg = DEFAULT_PIPELINE_CONFIG
    init_root = Path(dataset_root).resolve() if dataset_root else (dataset_dir / profile.root_init).resolve()
    output_root = Path(out_dir).resolve() if out_dir else dataset_dir / "audit_folder"
    target_folder = resolve_audit_folder(folder, dataset_dir, profile, init_root)

    target_files = sorted(target_folder.glob("*.arc_data"))
    if not target_files:
        raise FileNotFoundError(f"No .arc_data files in {target_folder}")

    processor = Processor(profile, cfg)
    ref_files, ref_dirs = reference_files(init_root, target_folder, audit_cfg)
    ref_items = [(path, processor.load(path)) for path in ref_files]
    ref_items = [(path, item) for path, item in ref_items if item is not None]
    if not ref_items:
        raise RuntimeError("No valid reference spectra were found")

    target_items = [(path, processor.load(path)) for path in target_files]
    target_items = [(path, item) for path, item in target_items if item is not None]
    if not target_items:
        raise RuntimeError("No valid target spectra were found")

    ref_arr = np.vstack([item[0] for _, item in ref_items])
    target_arr = np.vstack([item[0] for _, item in target_items])
    rows, thresholds, ref_median, _, folder_median = score_reference_rows(
        [path for path, _ in target_items],
        target_arr,
        [item[1] for _, item in target_items],
        ref_arr,
        [item[1] for _, item in ref_items],
        audit_cfg,
    )
    csv_path, json_path, png_path = _write_outputs(
        output_root,
        target_folder,
        init_root,
        ref_dirs,
        rows,
        thresholds,
        no_plot,
        target_arr,
        ref_median,
        folder_median,
        cfg,
    )

    print(f"Folder: {target_folder.relative_to(init_root)}")
    print(f"Reference dirs: {', '.join(ref_dirs)}")
    print(f"Keep: {sum(row['decision'] == 'keep' for row in rows)}")
    print(f"Remove candidates: {sum(row['decision'] == 'remove' for row in rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    if png_path is not None:
        print(f"PNG: {png_path}")
    if thresholds["folder_warning"]:
        print(f"WARNING: folder median is far from references (corr={thresholds['folder_corr_ref']:.3f})")
    return {"rows": rows, "thresholds": thresholds, "csv": str(csv_path), "json": str(json_path), "png": str(png_path) if png_path else ""}


def build_parser():
    parser = argparse.ArgumentParser(description="审核小文件夹相对同属同前缀参考组的离群情况")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id")
    parser.add_argument("--folder", required=True, help="指定小文件夹，例如 Acinetobacter/AB01 或 AB01")
    parser.add_argument("--dataset-root", default=None, help="覆盖 init 根目录")
    parser.add_argument("--out-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--no-plot", action="store_true", help="不生成复核图")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    audit_folder(args.dataset, args.folder, dataset_root=args.dataset_root, out_dir=args.out_dir, no_plot=args.no_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

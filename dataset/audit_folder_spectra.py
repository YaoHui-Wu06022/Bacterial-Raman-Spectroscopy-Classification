"""Score single spectra in one init subfolder against same-prefix references.

Usage examples:
    python dataset/audit_folder_spectra.py 细菌 --folder Acinetobacter/AB01
    python dataset/audit_folder_spectra.py 细菌 --folder AB01
    python dataset/audit_folder_spectra.py 细菌 --folder E:/.../dataset/细菌/init/Acinetobacter/AB01

The script does not move or delete files. It writes a CSV/JSON summary and a
review plot so questionable spectra can be checked before manual cleanup.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_single_spectra import preprocess_spectrum_for_audit, robust_wave_stats, spectral_corr  # noqa: E402
from raman.data.build import DEFAULT_PIPELINE_CONFIG  # noqa: E402
from raman.data.profiles import get_dataset_dir, get_profile  # noqa: E402


# python dataset\audit_folder_spectra.py 细菌 --folder Acinetobacter/AB01

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="细菌", help="Dataset name or profile id, e.g. 细菌 / bacteria.")
    parser.add_argument("--folder", required=True, help="Folder to audit, e.g. Acinetobacter/AB01 or AB01.")
    parser.add_argument("--dataset-root", default=None, help="Override init root containing genus/folder data.")
    parser.add_argument("--out-dir", default=None, help="Override audit_folder output root.")
    parser.add_argument("--profile", default=None, help="Override dataset profile id used by preprocessing.")
    parser.add_argument("--min-ref-files", type=int, default=20, help="Minimum same-prefix reference files.")
    parser.add_argument("--no-plot", action="store_true", help="Skip review plot generation.")
    return parser.parse_args()


def prefix_of(name: str) -> str:
    match = re.match(r"([A-Za-z]+)", name)
    return match.group(1) if match else name


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def resolve_target_folder(folder_arg: str, dataset_root: Path) -> Path:
    direct = Path(folder_arg)
    candidates = []

    if direct.is_absolute() and direct.exists():
        return direct.resolve()
    if direct.exists():
        return direct.resolve()

    for rel_candidate in ((dataset_root / folder_arg).resolve(), (dataset_root.parent / folder_arg).resolve()):
        if rel_candidate.exists():
            return rel_candidate

    for folder in dataset_root.glob("*/*"):
        if folder.is_dir() and folder.name == folder_arg:
            candidates.append(folder.resolve())

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        joined = "\n".join(str(path) for path in candidates)
        raise ValueError(f"Folder name is ambiguous. Use Genus/Folder:\n{joined}")
    raise FileNotFoundError(f"Folder not found: {folder_arg}")


class Processor:
    def __init__(self, profile_id: str):
        self.cfg = DEFAULT_PIPELINE_CONFIG
        self.profile = get_profile(profile_id)
        self.wn_ref = self.cfg.build_wn_ref()
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


def reference_files(dataset_root: Path, target_folder: Path, min_ref_files: int) -> tuple[list[Path], list[str]]:
    rel = target_folder.relative_to(dataset_root)
    genus = rel.parts[0]
    folder_name = rel.parts[1]
    genus_root = dataset_root / genus
    target_prefix = prefix_of(folder_name)

    folders = [path for path in sorted(genus_root.iterdir()) if path.is_dir() and path != target_folder]
    same_prefix = [path for path in folders if prefix_of(path.name) == target_prefix]
    chosen = same_prefix if sum(len(list(path.glob("*.arc_data"))) for path in same_prefix) >= min_ref_files else folders

    files: list[Path] = []
    for folder in chosen:
        files.extend(sorted(folder.glob("*.arc_data")))
    return files, [folder.name for folder in chosen]


def build_rows(
    target_files: list[Path],
    target_arr: np.ndarray,
    target_stats: list[object],
    ref_arr: np.ndarray,
    ref_stats: list[object],
    ref_median: np.ndarray,
    ref_scale: np.ndarray,
) -> tuple[list[dict], dict]:
    folder_median = np.median(target_arr, axis=0)
    ref_corrs = np.array([spectral_corr(spec, ref_median) for spec in ref_arr])
    ref_bad6 = np.array([float(np.mean(np.abs((spec - ref_median) / ref_scale) > 6.0)) for spec in ref_arr])
    ref_rmse = np.array([float(np.sqrt(np.mean((spec - ref_median) ** 2))) for spec in ref_arr])

    corr_threshold = max(0.80, float(np.percentile(ref_corrs, 1) - 0.05))
    nearest_threshold = max(0.86, float(np.percentile(ref_corrs, 5) - 0.03))
    bad_threshold = max(0.04, float(np.percentile(ref_bad6, 99) + 0.02))
    rmse_threshold = max(0.75, float(np.percentile(ref_rmse, 99) * 1.35))

    ref_cosmic = np.array([int(getattr(stats, "total", 0)) for stats in ref_stats], dtype=float)
    cosmic_threshold = 120
    if ref_cosmic.size:
        cosmic_threshold = max(80, int(np.percentile(ref_cosmic, 99) + 30))

    folder_corrs = np.array([spectral_corr(spec, folder_median) for spec in target_arr])
    folder_threshold = max(0.85, float(np.percentile(folder_corrs, 10) - 0.05))

    folder_corr_ref = spectral_corr(folder_median, ref_median)
    folder_warning = folder_corr_ref < max(0.75, corr_threshold - 0.08)

    thresholds = {
        "corr_ref_min": corr_threshold,
        "nearest_ref_corr_min": nearest_threshold,
        "bad_ratio_z6_max": bad_threshold,
        "rmse_to_ref_max": rmse_threshold,
        "cosmic_total_max": cosmic_threshold,
        "corr_folder_min": folder_threshold,
        "folder_corr_ref": folder_corr_ref,
        "folder_warning": bool(folder_warning),
    }

    rows = []
    for path, spec, stats in zip(target_files, target_arr, target_stats):
        abs_dz = np.abs((spec - ref_median) / ref_scale)
        nearest = max(spectral_corr(spec, ref_spec) for ref_spec in ref_arr)
        row = {
            "file": path.name,
            "path": str(path),
            "corr_ref": spectral_corr(spec, ref_median),
            "nearest_ref_corr": nearest,
            "corr_folder": spectral_corr(spec, folder_median),
            "bad_ratio_z6": float(np.mean(abs_dz > 6.0)),
            "bad_ratio_z8": float(np.mean(abs_dz > 8.0)),
            "max_abs_z": float(np.max(abs_dz)),
            "rmse_to_ref": float(np.sqrt(np.mean((spec - ref_median) ** 2))),
            "cosmic_total": int(getattr(stats, "total", 0)),
            "cosmic_narrow": int(getattr(stats, "narrow", 0)),
            "cosmic_peak": int(getattr(stats, "peak", 0)),
            "cosmic_residual": int(getattr(stats, "residual", 0)),
        }

        reasons = []
        if row["corr_ref"] < corr_threshold and row["nearest_ref_corr"] < nearest_threshold:
            reasons.append("low_ref_similarity")
        if row["bad_ratio_z6"] > bad_threshold:
            reasons.append("many_pointwise_outliers")
        if row["rmse_to_ref"] > rmse_threshold:
            reasons.append("high_rmse_to_ref")
        if row["cosmic_total"] > cosmic_threshold:
            reasons.append("excessive_cosmic_cleanup")
        if row["corr_folder"] < folder_threshold:
            reasons.append("low_folder_similarity")

        row["decision"] = "remove" if len(reasons) >= 2 else "keep"
        row["reasons"] = ",".join(reasons)
        rows.append(row)

    return rows, thresholds


def write_outputs(
    out_root: Path,
    target_folder: Path,
    dataset_root: Path,
    ref_dirs: list[str],
    rows: list[dict],
    thresholds: dict,
    no_plot: bool,
    target_arr: np.ndarray,
    ref_median: np.ndarray,
    folder_median: np.ndarray,
) -> None:
    rel = target_folder.relative_to(dataset_root)
    out_dir = out_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scores.csv"
    json_path = out_dir / "summary.json"
    png_path = out_dir / "review.png"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "folder": str(rel).replace("\\", "/"),
        "target_count": len(rows),
        "reference_dirs": ref_dirs,
        "thresholds": thresholds,
        "remove_count": sum(row["decision"] == "remove" for row in rows),
        "keep_count": sum(row["decision"] == "keep" for row in rows),
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    if no_plot:
        print(f"CSV: {csv_path}")
        print(f"JSON: {json_path}")
        return

    cfg = DEFAULT_PIPELINE_CONFIG
    wn_full = cfg.build_wn_ref()
    keep_mask = np.ones_like(wn_full, dtype=bool)
    for band_min, band_max in cfg.bad_bands:
        keep_mask &= ~((wn_full >= band_min) & (wn_full <= band_max))
    wn = wn_full[keep_mask]

    plot_items = sorted(zip(rows, target_arr), key=lambda item: (item[0]["decision"] != "remove", item[0]["corr_ref"]))
    cols = 3
    nrows = int(np.ceil(len(plot_items) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(15, max(3, nrows * 2.4)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (row, spec) in zip(axes, plot_items):
        color = "#D62728" if row["decision"] == "remove" else "#1F77B4"
        ax.plot(wn, ref_median, color="#444444", lw=1.1, label="ref median")
        ax.plot(wn, folder_median, color="#0C7C59", lw=1.0, label="folder median")
        ax.plot(wn, spec, color=color, lw=0.9)
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
    fig.suptitle(f"{str(rel).replace(chr(92), '/')} single-spectrum review", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")


def main() -> int:
    args = parse_args()
    profile = get_profile(args.profile or args.dataset)
    dataset_dir = get_dataset_dir(profile, PROJECT_ROOT)
    dataset_root = resolve_path(args.dataset_root, PROJECT_ROOT) if args.dataset_root else (dataset_dir / profile.root_init).resolve()
    out_dir = resolve_path(args.out_dir, PROJECT_ROOT) if args.out_dir else dataset_dir / "audit_folder"
    target_folder = resolve_target_folder(args.folder, dataset_root)

    target_files = sorted(target_folder.glob("*.arc_data"))
    if not target_files:
        raise FileNotFoundError(f"No .arc_data files in {target_folder}")

    processor = Processor(profile.profile_id)
    ref_files, ref_dirs = reference_files(dataset_root, target_folder, args.min_ref_files)
    ref_items = [(path, processor.load(path)) for path in ref_files]
    ref_items = [(path, item) for path, item in ref_items if item is not None]
    if not ref_items:
        raise RuntimeError("No valid reference spectra were found.")

    target_items = [(path, processor.load(path)) for path in target_files]
    target_items = [(path, item) for path, item in target_items if item is not None]
    if not target_items:
        raise RuntimeError("No valid target spectra were found.")

    ref_arr = np.vstack([item[0] for _, item in ref_items])
    target_arr = np.vstack([item[0] for _, item in target_items])
    target_stats = [item[1] for _, item in target_items]
    ref_median, ref_scale = robust_wave_stats(ref_arr, min_scale=0.05, floor_fraction=0.25)
    folder_median = np.median(target_arr, axis=0)

    rows, thresholds = build_rows(
        [path for path, _ in target_items],
        target_arr,
        target_stats,
        ref_arr,
        [item[1] for _, item in ref_items],
        ref_median,
        ref_scale,
    )
    write_outputs(
        out_dir,
        target_folder,
        dataset_root,
        ref_dirs,
        rows,
        thresholds,
        args.no_plot,
        target_arr,
        ref_median,
        folder_median,
    )

    print(f"Folder: {target_folder.relative_to(dataset_root)}")
    print(f"Reference dirs: {', '.join(ref_dirs)}")
    print(f"Keep: {sum(row['decision'] == 'keep' for row in rows)}")
    print(f"Remove candidates: {sum(row['decision'] == 'remove' for row in rows)}")
    if thresholds["folder_warning"]:
        print(f"WARNING: folder median is far from references (corr={thresholds['folder_corr_ref']:.3f}).")
    for row in rows:
        if row["decision"] == "remove":
            print(
                f"  REMOVE {row['file']}: corr_ref={row['corr_ref']:.3f}, "
                f"corr_folder={row['corr_folder']:.3f}, bad6={row['bad_ratio_z6']:.3f}, "
                f"cosmic={row['cosmic_total']}, reasons={row['reasons']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""按属审核文件夹的重复测量稳定性与跨属谱形边界。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from raman.audit.common import preprocess_spectrum_for_audit
from raman.audit.pair_review import _plot_pair
from raman.pipeline import DEFAULT_PIPELINE_CONFIG
from raman.tool.dataset import resolve_dataset
from raman.tool.path import PROJECT_ROOT


PAIR_PATTERN = re.compile(r"^(.*)_(000|001)_shift_cos\.arc_data$", re.IGNORECASE)


def _corr(first: np.ndarray, second: np.ndarray) -> float:
    """计算两条标准化谱的 Pearson 相关性。"""
    first = first - first.mean()
    second = second - second.mean()
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-8)
    return float(first @ second / denominator)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """以 UTF-8 BOM 写出审核 CSV。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def _load_records(dataset_dir: Path):
    """读取 init 内全部可处理谱，并保留属、文件夹和处理后的谱。"""
    profile, _ = resolve_dataset(dataset_dir.name, PROJECT_ROOT)
    records = []
    for path in sorted((dataset_dir / "init").rglob("*.arc_data")):
        relative = path.relative_to(dataset_dir / "init")
        if len(relative.parts) < 3:
            continue
        payload = preprocess_spectrum_for_audit(path, profile, DEFAULT_PIPELINE_CONFIG)
        if payload.get("skip_reason"):
            continue
        records.append(
            {
                "path": path,
                "rel_path": relative.as_posix(),
                "genus": relative.parts[0],
                "folder": relative.parts[1],
                "z": np.asarray(payload["z"], dtype=np.float32),
            }
        )
    return records


def run_genus_review(
    dataset_key: str,
    genera: tuple[str, ...],
    pair_plots_per_folder: int = 2,
) -> Path:
    """生成目标属的文件夹稳定性、近邻边界和低一致性对图。"""
    if pair_plots_per_folder < 1:
        raise ValueError("每个文件夹至少绘制一对重复测量")
    _, dataset_dir = resolve_dataset(dataset_key, PROJECT_ROOT)
    records = _load_records(dataset_dir)
    by_folder: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in records:
        by_folder[(str(item["genus"]), str(item["folder"]))].append(item)
    representatives = {
        key: np.median(np.stack([np.asarray(item["z"]) for item in items]), axis=0)
        for key, items in by_folder.items()
    }

    pair_groups: dict[tuple[str, str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for item in records:
        if item["genus"] not in genera:
            continue
        match = PAIR_PATTERN.match(Path(item["path"]).name)
        if match is not None:
            pair_groups[(str(item["genus"]), str(item["folder"]), match.group(1))][match.group(2)] = item

    pair_rows = []
    pair_items: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for (genus, folder, cell), sides in sorted(pair_groups.items()):
        if set(sides) != {"000", "001"}:
            continue
        first, second = sides["000"], sides["001"]
        item = {
            "genus": genus,
            "folder": folder,
            "cell": cell,
            "first_path": first["path"],
            "second_path": second["path"],
            "first": first["z"],
            "second": second["z"],
            "corr": _corr(np.asarray(first["z"]), np.asarray(second["z"])),
        }
        pair_items[(genus, folder)].append(item)
        pair_rows.append(
            {
                "genus": genus,
                "folder": folder,
                "cell": cell,
                "pair_corr": f"{item['corr']:.6f}",
                "first_rel_path": first["rel_path"],
                "second_rel_path": second["rel_path"],
            }
        )

    summary_rows = []
    for (genus, folder), representative in sorted(representatives.items()):
        if genus not in genera:
            continue
        same = [
            (other_folder, _corr(representative, other_rep))
            for (other_genus, other_folder), other_rep in representatives.items()
            if other_genus == genus and other_folder != folder
        ]
        foreign = [
            (other_genus, other_folder, _corr(representative, other_rep))
            for (other_genus, other_folder), other_rep in representatives.items()
            if other_genus != genus
        ]
        nearest_same_folder, nearest_same_corr = max(same, key=lambda value: value[1]) if same else ("", float("nan"))
        nearest_foreign_genus, nearest_foreign_folder, nearest_foreign_corr = max(foreign, key=lambda value: value[2])
        pair_values = [float(item["corr"]) for item in pair_items.get((genus, folder), [])]
        summary_rows.append(
            {
                "genus": genus,
                "folder": folder,
                "spectra": len(by_folder[(genus, folder)]),
                "pair_count": len(pair_values),
                "pair_corr_median": f"{np.median(pair_values):.6f}" if pair_values else "",
                "pair_corr_min": f"{np.min(pair_values):.6f}" if pair_values else "",
                "nearest_same_folder": nearest_same_folder,
                "nearest_same_corr": f"{nearest_same_corr:.6f}",
                "nearest_foreign_genus": nearest_foreign_genus,
                "nearest_foreign_folder": nearest_foreign_folder,
                "nearest_foreign_corr": f"{nearest_foreign_corr:.6f}",
                "same_minus_foreign": f"{nearest_same_corr - nearest_foreign_corr:.6f}",
            }
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = dataset_dir / "audit_runs" / f"{stamp}_genus_review"
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(out_dir / "folder_summary.csv", summary_rows)
    _write_csv(out_dir / "pair_scores.csv", sorted(pair_rows, key=lambda row: (row["folder"], float(row["pair_corr"]))))
    figures = 0
    for (genus, folder), items in sorted(pair_items.items()):
        for rank, item in enumerate(sorted(items, key=lambda value: value["corr"])[:pair_plots_per_folder], start=1):
            output = out_dir / "figures" / genus / folder / f"{rank:02d}_{item['cell']}.png"
            _plot_pair(item, output, rank)
            figures += 1
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "dataset": dataset_key,
                "genera": list(genera),
                "records": len(records),
                "target_folders": len(summary_rows),
                "pair_rows": len(pair_rows),
                "figures": figures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def main(argv=None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="按属审核批次稳定性与跨属谱形边界")
    parser.add_argument("--dataset", default="GN", help="数据集 profile id")
    parser.add_argument("--genera", nargs="+", default=["Escherichia", "Citrobacter"], help="待审核属名")
    parser.add_argument("--pair-plots-per-folder", type=int, default=2, help="每个文件夹绘制的低一致性对数")
    args = parser.parse_args(argv)
    print(run_genus_review(args.dataset, tuple(args.genera), args.pair_plots_per_folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

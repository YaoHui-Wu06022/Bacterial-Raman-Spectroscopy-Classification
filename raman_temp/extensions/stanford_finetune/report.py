"""Stanford 预训练和迁移报告写入。"""

from __future__ import annotations

import json
from pathlib import Path


def write_pretrain_report(experiment_dir: Path, level_name: str, class_names) -> Path:
    """记录 Stanford 预训练的固定输入约束和类别摘要。"""
    output_path = experiment_dir / "pretrain_report.json"
    output_path.write_text(
        json.dumps(
            {
                "profile_id": "Stanford",
                "level_name": level_name,
                "input_grid_mode": "stanford_transfer",
                "norm_method": "minmax",
                "class_names": list(class_names),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_transfer_reports(experiment_dir: Path, payload: dict, reports: dict[str, dict]) -> Path:
    """在实验根和实际 run 写入来源、加载和冻结范围报告。"""
    root_path = experiment_dir / "transfer_report.json"
    root_path.write_text(
        json.dumps(payload | {"models": reports}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for model_tag, values in reports.items():
        candidates = list(experiment_dir.rglob(f"{model_tag}_model.pt"))
        for model_path in candidates:
            run_path = model_path.parent
            (run_path / "transfer_report.json").write_text(
                json.dumps(values, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return root_path

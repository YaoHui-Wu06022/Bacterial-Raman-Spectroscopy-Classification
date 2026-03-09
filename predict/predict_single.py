# -*- coding: utf-8 -*-
# ============================================================
#  预测单个文件夹（基于实验目录）
# ============================================================

import os
import re
import torch
from predict_core import load_predictor, predict_one
from raman.data_paths import resolve_dataset_stage

# 项目根目录解析（支持子目录运行）
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path):
    if path is None:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(BASE_DIR, path))


def get_cell_number(fname):
    """
    从文件名中提取 cell 后的数字编号
    兼容 cell / Cell / CELL
    用于排序，保证 batch 输出的物理顺序一致
    """
    m = re.search(r"cell(\d+)", fname, re.IGNORECASE)
    if m is not None:
        return int(m.group(1))
    # fallback：异常命名文件统一放到最后
    return int(1e9)


if __name__ == "__main__":

    # 指定实验目录（必须是训练输出目录）
    # 里面应包含：
    #   - config.yaml
    #   - {TASK}_model.pt
    EXP_DIR = resolve_path("output/耐药菌/20260129_052515")
    # 手动设置预测层级（None 则使用 config）
    PREDICT_LEVEL = "level_3"

    # 待预测的文件夹（包含 .arc_data）
    dataset_root = resolve_path("dataset/耐药菌")
    folder_name = "10CS"
    if not folder_name.strip():
        raise ValueError("Please set folder_path to a valid folder path.")
    folder_path = os.path.join(
        os.fspath(
            resolve_dataset_stage(
                dataset_root,
                stage="predict_input",
                project_root=BASE_DIR,
                must_exist=True,
            )
        ),
        folder_name
    )
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    top_k = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建 predictor（基于实验目录）
    predictor = load_predictor(EXP_DIR, device, predict_level=PREDICT_LEVEL)

    # 可选：人工指定上层遮罩（例如在 level_1 先验已知大类）
    # 例如：{"level_1": ["baoman", "dachang"]}
    # 也支持索引：{"level_1": [0, 2]}
    MANUAL_PARENT_MASK = {
        "level_1":["feike"],
        "level_2":["feike/ESBL"]
    }

    # 输出目录（默认写到实验目录下）
    output_dir = resolve_path(os.path.join(EXP_DIR, f"predict_results_{PREDICT_LEVEL}"))
    os.makedirs(output_dir, exist_ok=True)

    folder_name = os.path.basename(folder_path.rstrip("/"))
    output_file_txt = os.path.join(output_dir, f"{folder_name}_file.txt")

    files = sorted(
        (f for f in os.listdir(folder_path) if f.endswith(".arc_data")),
        key=get_cell_number
    )

    if not files:
        raise FileNotFoundError("No .arc_data files found in the folder.")

    # ---------------- file-level buffers ----------------
    summary_counter = {}
    details_lines = []

    for fname in files:
        fp = os.path.join(folder_path, fname)

        # ===== file-level 预测 =====
        results = predict_one(fp, predictor, top_k=top_k, parent_mask=MANUAL_PARENT_MASK)

        # ===== 统计 file-level =====
        top1 = results[0]
        summary_counter[top1["label"]] = summary_counter.get(
            top1["label"], 0
        ) + 1

        details_lines.append(f"########## File: {fname} ##########\n")
        details_lines.append(
            f"Top-1 => {top1['label']} ({top1['prob'] * 100:.2f}%)\n"
        )
        details_lines.append("Top-k predictions:\n")

        for i, r in enumerate(results, 1):
            details_lines.append(
                f"   {i}) {r['label']:20s}  {r['prob'] * 100:.2f}%\n"
            )

        details_lines.append("\n===============================================\n\n")

    # ======================================================
    # FILE-LEVEL SUMMARY
    # ======================================================
    summary_lines = []
    summary_lines.append("===== FILE-LEVEL SUMMARY =====\n\n")

    for k, v in sorted(summary_counter.items(), key=lambda x: -x[1]):
        summary_lines.append(f"{k:20s} : {v}\n")

    summary_lines.append("\n===============================================\n\n")

    # ======================================================
    # SAVE
    # ======================================================
    with open(output_file_txt, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
        f.writelines(details_lines)

    print(f"[Saved] file-level => {output_file_txt}")

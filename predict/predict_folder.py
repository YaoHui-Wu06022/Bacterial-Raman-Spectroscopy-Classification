# ============================================================
#  预测文件夹（批量，基于实验目录）
# ============================================================

import os
import torch
from tqdm import tqdm
from predict_core import load_predictor, predict_one
import re

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


# ============================================================
# Batch prediction for one folder
# ============================================================
def predict_folder(folder_path, output_dir, predictor, top_k=3, parent_mask=None):

    folder_path = resolve_path(folder_path)
    output_dir = resolve_path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    folder_name = os.path.basename(folder_path.rstrip("/"))
    output_file_txt = os.path.join(output_dir, f"{folder_name}_file.txt")

    files = sorted(
        (f for f in os.listdir(folder_path) if f.endswith(".arc_data")),
        key=get_cell_number
    )

    if not files:
        print("No .arc_data files found.")
        return

    # ---------------- file-level buffers ----------------
    summary_counter = {}
    details_lines = []

    for fname in tqdm(files, desc="Predicting"):
        fp = os.path.join(folder_path, fname)

        # ===== file-level 预测 =====
        results = predict_one(fp, predictor, top_k=top_k, parent_mask=parent_mask)

        # ===== 统计 file-level =====
        top1 = results[0]
        summary_counter[top1["label"]] = summary_counter.get(
            top1["label"], 0
        ) + 1

        details_lines.append(f"########## File: {fname} ##########\n")
        details_lines.append(
            f"Top-1 → {top1['label']} ({top1['prob'] * 100:.2f}%)\n"
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

    print(f"[Saved] file-level → {output_file_txt}")
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # ========================================================
    # 指定实验输出目录（必须包含 config_.yaml + *_model.pt）
    # ========================================================
    EXP_DIR = resolve_path("output_耐药菌/20260302_021659")
    # 手动设置预测层级（None 则使用 config）
    PREDICT_LEVEL = "level_2"

    # 可选：人工指定上层遮罩（例如在 level_1 先验已知大类）
    # 例如：{"level_1": ["baoman", "dachang"]}
    # 也支持索引：{"level_1": [0, 2]}
    MANUAL_PARENT_MASK = {
        "level_1":["feike"],
    }
    # MANUAL_PARENT_MASK = None

    # 待预测数据根目录
    PREDICT_ROOT = resolve_path("dataset_test_耐药菌")
    if not os.path.isdir(PREDICT_ROOT):
        raise FileNotFoundError(
            f"Predict root not found: {PREDICT_ROOT}. Please check the path."
        )

    TOP_K = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================================================
    # 构建 predictor
    # ========================================================
    predictor = load_predictor(EXP_DIR, device, predict_level=PREDICT_LEVEL)

    # ========================================================
    # 输出目录：统一放到实验目录下
    # ========================================================
    out_root = os.path.join(EXP_DIR, f"predict_results_{PREDICT_LEVEL}")
    os.makedirs(out_root, exist_ok=True)

    subs = [
        d for d in os.listdir(PREDICT_ROOT)
        if os.path.isdir(os.path.join(PREDICT_ROOT, d))
    ]

    if not subs:
        print("No sub-folders found in predict root.")
        exit(0)

    print("\n>>> Batch prediction started ...\n")

    for sub in subs:
        inp = os.path.join(PREDICT_ROOT, sub)
        out = os.path.join(out_root, sub)

        print("\n---------------------------------------")
        print(f"Input folder : {inp}")
        print(f"Output file  : {out}")
        print("---------------------------------------")

        predict_folder(inp, out, predictor, top_k=TOP_K, parent_mask=MANUAL_PARENT_MASK)

    print("\n===== All folders processed =====\n")

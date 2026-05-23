import os
import re
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path):
    """把相对路径解析到项目根目录"""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def get_cell_number(fname):
    """从文件名中提取 cell 编号用于稳定排序"""
    match = re.search(r"cell(\d+)", fname, re.IGNORECASE)
    return int(match.group(1)) if match else int(1e9)


def iter_predict_folders(predict_root, one_folder=None):
    """列出本次需要预测的测试文件夹"""
    predict_root = Path(resolve_path(predict_root))
    if one_folder:
        folder_path = Path(one_folder)
        if not folder_path.is_absolute():
            folder_path = predict_root / folder_path
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Input folder not found: {folder_path}")
        return [folder_path]

    return sorted(path for path in predict_root.iterdir() if path.is_dir())


def list_arc_files(folder_path):
    """列出单个文件夹中的光谱文件"""
    folder_path = Path(folder_path)
    return sorted(
        (path for path in folder_path.iterdir() if path.suffix.lower() == ".arc_data"),
        key=lambda path: get_cell_number(path.name),
    )


def format_prediction_report(folder_name, predictions, folder_summary=None):
    """生成逐谱 top-k 预测文本"""
    counter = Counter(item["top1_label"] for item in predictions)
    lines = []
    if folder_summary:
        lines.extend(
            [
                "===== FOLDER SUMMARY =====\n\n",
                f"Expected label      : {folder_summary['expected_label']}\n",
                f"Expected in model   : {folder_summary['expected_in_model']}\n",
                f"Majority prediction : {folder_summary['predicted_label']}\n",
                f"Correct spectra     : {folder_summary['correct_count']}/{folder_summary['total_count']} ({folder_summary['correct_ratio'] * 100:.2f}%)\n",
                f"Folder correct      : {folder_summary['folder_correct']}\n",
                "\n===============================================\n\n",
            ]
        )

    lines.append("===== FILE-LEVEL SUMMARY =====\n\n")
    for label, count in counter.most_common():
        lines.append(f"{label:20s} : {count}\n")
    lines.append("\n===============================================\n\n")

    for item in predictions:
        lines.append(f"########## File: {item['file']} ##########\n")
        top1 = item["results"][0]
        lines.append(f"Top-1 -> {top1['label']} ({top1['prob'] * 100:.2f}%)\n")
        lines.append("Top-k predictions:\n")
        for idx, result in enumerate(item["results"], 1):
            lines.append(f"   {idx}) {result['label']:20s}  {result['prob'] * 100:.2f}%\n")
        lines.append("\n===============================================\n\n")
    return lines


def predict_directory(folder_path, output_dir, predictor, top_k=3, parent_mask=None):
    """预测单个文件夹并写出旧版逐谱文本"""
    from tqdm import tqdm
    from raman.infer.core import predict_one

    folder_path = Path(resolve_path(folder_path))
    output_dir = Path(resolve_path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_arc_files(folder_path)
    if not files:
        print("No .arc_data files found.")
        return []

    predictions = []
    for path in tqdm(files, desc=f"Predicting {folder_path.name}"):
        results = predict_one(path, predictor, top_k=top_k, parent_mask=parent_mask)
        predictions.append(
            {
                "file": path.name,
                "path": path,
                "results": results,
                "top1_label": results[0]["label"],
            }
        )

    output_file = output_dir / f"{folder_path.name}_file.txt"
    output_file.write_text(
        "".join(format_prediction_report(folder_path.name, predictions)),
        encoding="utf-8",
    )
    print(f"[Saved] file-level -> {os.fspath(output_file)}")
    return predictions

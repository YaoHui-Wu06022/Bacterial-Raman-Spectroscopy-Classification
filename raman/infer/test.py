from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from raman.tool.dataset import dataset_bundle_root, resolve_dataset_stage
from raman.tool.hierarchy import normalize_level_name
from raman.tool.naming import test_folder_prefix
from raman.tool.path import PROJECT_ROOT, resolve_project_path
from raman.tool.spectrum import expected_wavenumbers, get_config_bad_bands
from raman.eval.experiment import (
    collect_used_runs,
    resolve_result_dir,
    write_used_runs,
)
from raman.infer.folder import (
    format_prediction_report,
    iter_predict_folders,
    list_arc_files,
)
from raman.infer.labels import (
    build_expected_lookup_from_meta,
    folder_summary,
    write_summary,
)
from raman.infer.spectra import (
    build_train_mean_bank,
    plot_spectra,
    preprocess_with_config_mask,
)


def _resolve_test_root(config, override):
    """解析本次独立测试的已处理 test 目录"""
    if override:
        return resolve_project_path(override)
    dataset_root = dataset_bundle_root(resolve_project_path(config.dataset_root))
    return resolve_dataset_stage(
        dataset_root,
        stage="test",
        project_root=PROJECT_ROOT,
        must_exist=True,
    )


def _load_transfer_skip_lookup(manifest_path):
    """读取测试样本迁移 manifest，返回需要跳过的源文件"""
    if not manifest_path:
        return {}
    path = resolve_project_path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transfer manifest not found: {path}")

    lookup: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            source_folder = (row.get("source_folder") or "").strip()
            source_file = (row.get("source_file") or "").strip()
            if source_folder and source_file:
                lookup.setdefault(source_folder, set()).add(source_file)
    return lookup


def _predict_folder(folder_path, predictor, top_k, skip_lookup=None):
    """预测单个测试文件夹并收集绘图用谱线"""
    from tqdm import tqdm
    from raman.infer.core import predict_tensor

    predictions = []
    signals = []
    skipped = []
    skip_files = (skip_lookup or {}).get(folder_path.name, set())
    for path in tqdm(list_arc_files(folder_path), desc=f"Predicting {folder_path.name}", unit="spectrum"):
        if path.name in skip_files:
            skipped.append(path.name)
            continue
        x = preprocess_with_config_mask(path, predictor["preprocessor"], predictor["config"])
        results = predict_tensor(x, predictor, top_k=top_k)
        signal = x[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        predictions.append(
            {
                "file": path.name,
                "path": path,
                "results": results,
                "top1_label": results[0]["label"],
            }
        )
        signals.append(signal)
    return predictions, np.stack(signals, axis=0) if signals else np.empty((0, 0), dtype=np.float32), skipped


def _resolve_target_classes(predictor, input_context, level_name, runtime, class_names):
    """解析单 parent run 的类别范围"""
    if not (
        input_context.is_single_run
        and input_context.input_level == level_name
        and input_context.input_parent_idx is not None
    ):
        return None, class_names

    target_parent_idx = int(input_context.input_parent_idx)
    runtime.ensure_parent_models(level_name)
    entry = runtime.parent_models.get(level_name, {}).get(target_parent_idx, {})
    child_ids = [int(item) for item in entry.get("child_ids", [])]
    if child_ids and not predictor.get("class_names"):
        class_names = [class_names[child_id] for child_id in child_ids]
    return target_parent_idx, class_names


def _reset_result_dir(exp_dir, level_name, predictor, input_context, runtime, target_parent_idx):
    """解析并重建本次 infer 输出目录"""
    out_root = resolve_result_dir(
        exp_dir,
        "infer",
        level_name,
        input_context=input_context,
        runtime=runtime,
        target_parent_idx=target_parent_idx,
        prefer_run_dir=(input_context.is_single_run or len(predictor["level_order"]) == 1),
    )
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def _write_used_runs(out_root, exp_dir, predictor, runtime, level_name, target_parent_idx):
    """记录当前 infer 实际使用的模型 run"""
    used_runs = collect_used_runs(
        exp_dir,
        runtime,
        level_order=predictor["level_order"],
        target_level=level_name,
        target_parent_idx=target_parent_idx,
    )
    write_used_runs(
        out_root,
        mode="single_run" if out_root.parent.name.startswith("run_") else "cascade",
        target_level=level_name,
        target_parent=target_parent_idx,
        runs=used_runs,
    )


def _process_folder(
    folder_path,
    *,
    predictor,
    class_names,
    expected_lookup,
    evaluate,
    top_k,
    skip_lookup,
    out_root,
    wavenumbers,
    train_mean_bank,
    config,
):
    """预测一个测试文件夹，并写出逐谱文本和谱图"""
    folder_path = Path(folder_path)
    expected_prefix = test_folder_prefix(folder_path.name)
    expected_label = expected_lookup.get(expected_prefix) if evaluate else None
    if evaluate and expected_label not in set(class_names):
        print(
            f"[Skip] {folder_path.name}: expected label "
            f"{expected_label or expected_prefix} not in model classes"
        )
        return None, []

    predictions, signals, skipped = _predict_folder(
        folder_path,
        predictor,
        top_k,
        skip_lookup=skip_lookup,
    )
    skipped_rows = [f"{folder_path.name}/{item}" for item in skipped]
    if not predictions:
        if skipped:
            print(f"[Skip] {folder_path.name}: all spectra were transferred into training")
        return None, skipped_rows

    row = folder_summary(folder_path.name, expected_label, class_names, predictions)
    folder_out = out_root / folder_path.name
    folder_out.mkdir(parents=True, exist_ok=True)
    report_lines = format_prediction_report(
        folder_path.name,
        predictions,
        row if evaluate else None,
    )
    (folder_out / f"{folder_path.name}_file.txt").write_text("".join(report_lines), encoding="utf-8")

    plot_spectra(
        folder_out / "spectra.png",
        folder_path.name,
        signals,
        wavenumbers,
        row["expected_label"] if evaluate else None,
        row["predicted_label"],
        train_mean_bank,
        get_config_bad_bands(config),
    )
    return row, skipped_rows


def run_independent_test(
    exp_dir,
    level,
    test_root=None,
    folder=None,
    top_k=3,
    device=None,
    evaluate=True,
    plot_train_mean=False,
    skip_transferred=False,
    transfer_manifest=None,
):
    """运行独立测试集推理并输出逐谱明细、谱线图和汇总报表"""
    import torch
    from raman.infer.core import load_predictor

    level_name = normalize_level_name(level)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_predictor(exp_dir, device, predict_level=level_name)
    input_context = predictor["input_context"]
    exp_dir = Path(input_context.exp_dir)
    config = predictor["config"]
    runtime = predictor["runtime"]
    class_names = list(predictor.get("class_names") or runtime.class_names_by_level.get(level_name, []))
    target_parent_idx, class_names = _resolve_target_classes(
        predictor,
        input_context,
        level_name,
        runtime,
        class_names,
    )

    test_root = Path(_resolve_test_root(config, test_root))
    expected_lookup = (
        build_expected_lookup_from_meta(predictor["meta"], level_name)
        if evaluate
        else {}
    )
    train_mean_bank = (
        build_train_mean_bank(
            exp_dir,
            config.dataset_root,
            level_name,
            predictor["preprocessor"],
            config,
        )
        if plot_train_mean
        else {}
    )
    wavenumbers = expected_wavenumbers(config)

    out_root = _reset_result_dir(
        exp_dir,
        level_name,
        predictor,
        input_context,
        runtime,
        target_parent_idx,
    )
    _write_used_runs(out_root, exp_dir, predictor, runtime, level_name, target_parent_idx)

    folders = iter_predict_folders(test_root, folder)
    if not folders:
        raise FileNotFoundError(f"No test folders found under {test_root}")

    summary_rows = []
    skipped_rows = []
    skip_lookup = _load_transfer_skip_lookup(transfer_manifest) if skip_transferred else {}
    for folder_path in folders:
        row, skipped = _process_folder(
            folder_path,
            predictor=predictor,
            class_names=class_names,
            expected_lookup=expected_lookup,
            evaluate=evaluate,
            top_k=top_k,
            skip_lookup=skip_lookup,
            out_root=out_root,
            wavenumbers=wavenumbers,
            train_mean_bank=train_mean_bank,
            config=config,
        )
        if row is not None:
            summary_rows.append(row)
        skipped_rows.extend(skipped)

    write_summary(out_root / "summary.txt", summary_rows, evaluate=evaluate)
    if skipped_rows:
        (out_root / "skipped_transferred_samples.txt").write_text(
            "\n".join(skipped_rows) + "\n",
            encoding="utf-8",
        )
    print(f"[Saved] independent test results -> {out_root}")
    return out_root


def build_parser():
    """构建独立测试参数解析器"""
    parser = argparse.ArgumentParser(description="Run independent Raman test inference")
    parser.add_argument("--exp-dir", required=True, help="模型输出目录")
    parser.add_argument("--level", required=True, help="预测业务层级，例如 level_1、level1 或 1")
    parser.add_argument("--test-root", default=None, help="覆盖默认 test 目录")
    parser.add_argument("--folder", default=None, help="只运行某个测试文件夹")
    parser.add_argument("--top-k", type=int, default=3, help="逐谱输出 top-k 数量")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    parser.add_argument("--no-eval", action="store_true", help="只做预测，不按文件夹名称评估真值")
    parser.add_argument("--plot-train-mean", action="store_true", help="在谱图中叠加训练集均值曲线")
    parser.add_argument("--skip-transferred", action="store_true", help="跳过已迁移进训练集的测试谱")
    parser.add_argument("--transfer-manifest", default="dataset/测试菌/test_transfer_manifest.csv", help="测试谱迁移 manifest")
    return parser


def main(argv=None):
    """执行独立测试命令"""
    args = build_parser().parse_args(argv)
    import torch

    device = torch.device("cpu") if args.cpu else None
    run_independent_test(
        args.exp_dir,
        args.level,
        test_root=args.test_root,
        folder=args.folder,
        top_k=args.top_k,
        device=device,
        evaluate=not args.no_eval,
        plot_train_mean=args.plot_train_mean,
        skip_transferred=args.skip_transferred,
        transfer_manifest=args.transfer_manifest,
    )
    return 0

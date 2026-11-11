"""按层级拼接多个模型槽位的验证与测试结果。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ramanv2.common.device import resolve_device
from ramanv2.common.hierarchy import load_hierarchy_meta, normalize_level_name
from ramanv2.common.paths import relpath
from ramanv2.core.run_context import (
    CASCADE_MODE_DIR_NAME,
    HIERARCHY_META_FILE_NAME,
    LEVEL_SUMMARY_DIR_NAME,
    METRICS_FILE_NAME,
    MODEL_FILE_NAME,
    PARENT_KNOWN_MODE_DIR_NAME,
    TEST_RESULT_DIR_NAME,
    VAL_RESULT_DIR_NAME,
)
from ramanv2.data.dataset_io import load_arc_intensity
from ramanv2.evaluation.context import (
    EvaluationContext,
    load_evaluation_context,
    resolve_experiment_dir,
    resolve_level_name,
)
from ramanv2.evaluation.metrics_report import write_model_report
from ramanv2.evaluation.summary_text import build_result_summary
from ramanv2.evaluation.independent_test import (
    TEST_SAMPLE_CSV_NAME,
    compute_folder_metrics,
    resolve_preprocessed_test_dir,
)
from ramanv2.inference.spectrum_input import build_inference_preprocessor
from ramanv2.modeling.predictor import load_predictor
from ramanv2.modeling.run_manifest import (
    find_model_slot_dirs,
    has_model_path,
    resolve_entry_path,
)


VAL_SAMPLE_CSV_NAME = "val_eval_results.csv"
_KIND_SPECS = (
    ("val_result", VAL_RESULT_DIR_NAME, VAL_SAMPLE_CSV_NAME, False, False),
    ("test_result", TEST_RESULT_DIR_NAME, TEST_SAMPLE_CSV_NAME, True, True),
)


def assert_multi_model_level(
    source_dir: Path | str,
    level_name: str,
    summary_label: str,
) -> tuple[Path, dict[str, Any], str]:
    """确认层级存在父类模型槽位，且不是全局模型层级。"""
    experiment_dir = resolve_experiment_dir(source_dir)
    meta = load_hierarchy_meta(experiment_dir / HIERARCHY_META_FILE_NAME)
    if meta is None:
        raise FileNotFoundError(
            f"缺少 {HIERARCHY_META_FILE_NAME}：{experiment_dir}"
        )
    level = normalize_level_name(level_name)
    global_entry = (meta.get("level_models") or {}).get(level)
    if has_model_path(global_entry):
        raise ValueError(f"{level} 是全局模型层级，不需要{summary_label}")
    slots = find_model_slot_dirs(experiment_dir / level)
    if not slots:
        raise ValueError(f"{level} 没有父类模型槽位，无法生成{summary_label}")
    return experiment_dir, meta, level


def summarize_level_results(
    source_dir: Path | str,
    level_name: str,
    device: str | None = None,
) -> dict[str, Any]:
    """生成已知上层与从头级联两种模式的验证/测试汇总。"""
    assert_multi_model_level(source_dir, level_name, "层级汇总")
    context = load_evaluation_context(source_dir)
    level_name = resolve_level_name(context, level_name)
    level_dir = context.experiment_dir / level_name
    target_device = resolve_device(device)
    class_names = context.dataset_index.get_class_names(level_name)
    predictor_cache: dict[Path, tuple[Any, Any]] = {}
    summary_root = level_dir / LEVEL_SUMMARY_DIR_NAME
    parent_known_root = summary_root / PARENT_KNOWN_MODE_DIR_NAME
    for output_name, result_dir_name, sample_csv_name, folder_enable, covered_enable in _KIND_SPECS:
        frames, sources = _collect_level_frames(
            context,
            level_name,
            result_dir_name,
            sample_csv_name,
            target_device,
            predictor_cache,
        )
        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            raise RuntimeError(f"{level_name} 没有可汇总的 {result_dir_name} 结果")
        paths = combined["path"].astype(str).tolist()
        labels = combined["label_true"].astype(int).tolist()
        predictions = combined["label_pred"].astype(int).tolist()
        extra_summary: dict[str, Any] = {"sources": sources}
        if folder_enable:
            folders = [Path(path).parent.name for path in paths]
            extra_summary["folder"] = compute_folder_metrics(
                folders,
                labels,
                predictions,
                class_names,
            )
        write_model_report(
            parent_known_root / output_name,
            class_names,
            paths,
            labels,
            predictions,
            sample_name=sample_csv_name,
            extra_summary=extra_summary,
            metric_class_ids=sorted(set(labels)) if covered_enable else None,
        )

    from ramanv2.inference.folder_inference import (
        merge_level_test_inference_summary,
        resolve_level_inference_dir,
        run_experiment_inference,
    )

    test_dir = resolve_preprocessed_test_dir(
        context.config.dataset.profile_id,
        dataset_id=context.config.dataset.dataset_id,
    )
    cascade_results = run_experiment_inference(
        context.experiment_dir,
        test_dir,
        (level_name,),
        device=target_device,
        seed=context.config.execution.seed,
    )
    if level_name not in cascade_results:
        raise RuntimeError(f"{level_name} 的级联模型未补齐，无法生成 cascade 结果")
    cascade_root = summary_root / CASCADE_MODE_DIR_NAME
    inference_dir = resolve_level_inference_dir(
        context.experiment_dir,
        context.meta,
        level_name,
    )
    _write_cascade_test_report(
        context.meta,
        level_name,
        test_dir,
        inference_dir,
        cascade_root / TEST_RESULT_DIR_NAME,
    )
    _write_cascade_val_report(
        context,
        level_name,
        target_device,
        cascade_root / VAL_RESULT_DIR_NAME,
    )
    merge_level_test_inference_summary(context.experiment_dir, context.meta, level_name)
    for result_dir, mode, title in (
        (parent_known_root / VAL_RESULT_DIR_NAME, PARENT_KNOWN_MODE_DIR_NAME, "验证集评估汇总"),
        (parent_known_root / TEST_RESULT_DIR_NAME, PARENT_KNOWN_MODE_DIR_NAME, "独立测试集评估汇总"),
        (cascade_root / VAL_RESULT_DIR_NAME, CASCADE_MODE_DIR_NAME, "验证集评估汇总"),
    ):
        metrics_path = result_dir / METRICS_FILE_NAME
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            (result_dir / "summary.txt").write_text(
                build_result_summary(metrics, level_name, mode, title=title),
                encoding="utf-8",
            )
    return {
        "level_name": level_name,
        "level_summary_dir": str(summary_root),
        "results": {
            PARENT_KNOWN_MODE_DIR_NAME: str(parent_known_root),
            CASCADE_MODE_DIR_NAME: str(cascade_root),
        },
    }


def _write_cascade_test_report(
    meta: dict[str, Any],
    level_name: str,
    test_dir: Path,
    inference_dir: Path,
    output_dir: Path,
) -> None:
    """把级联推理的逐文件 top-1 写成标准 test_result 指标。"""
    payload = json.loads((inference_dir / "summary.json").read_text(encoding="utf-8"))
    class_names = list(meta["class_names_by_level"][level_name])
    name_to_id = {name: index for index, name in enumerate(class_names)}
    paths: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    for row in payload.get("rows") or []:
        expected_label = str(row.get("expected_label", ""))
        if expected_label not in name_to_id:
            continue
        folder = str(row.get("folder", ""))
        for item in row.get("file_predictions") or []:
            values = item.get("predictions") or []
            if not values:
                continue
            class_id = values[0].get("class_id")
            if class_id is None:
                continue
            paths.append(str(test_dir / folder / str(item.get("file", ""))))
            labels.append(name_to_id[expected_label])
            predictions.append(int(class_id))
    if not paths:
        raise RuntimeError(f"{level_name} 的级联测试结果没有可写入的预测")
    folders = [Path(path).parent.name for path in paths]
    folder_metrics = compute_folder_metrics(folders, labels, predictions, class_names)
    write_model_report(
        output_dir,
        class_names,
        paths,
        labels,
        predictions,
        sample_name=TEST_SAMPLE_CSV_NAME,
        extra_summary={
            "sources": [{"kind": "cascade", "level_name": level_name, "sample_count": len(paths)}],
            "folder": folder_metrics,
        },
        metric_class_ids=sorted(set(labels)),
    )


def _write_cascade_val_report(
    context: EvaluationContext,
    level_name: str,
    device: Any,
    output_dir: Path,
) -> None:
    """对验证集执行从顶层到本层的级联预测并写标准 val_result。"""
    predictor = load_predictor(context.experiment_dir, device, level_name)
    preprocessor = build_inference_preprocessor(context.input_spec, predictor.device)
    class_names = context.dataset_index.get_class_names(level_name)
    name_to_id = {name: index for index, name in enumerate(class_names)}
    paths: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    for raw_index in context.validation_indices:
        index = int(raw_index)
        inputs = preprocessor.preprocess_intensity(context.dataset_index.get_raw_intensity(index))
        prediction = predictor.predict_tensor(inputs, top_k=1)[0]
        true_key = context.dataset_index.get_level_key(index, level_name)
        if true_key not in name_to_id:
            raise ValueError(f"验证集真实标签不在目标层级类别中：{true_key}")
        paths.append(str(context.dataset_index.samples[index]))
        labels.append(name_to_id[true_key])
        predictions.append(int(prediction.class_id))
    if not paths:
        raise RuntimeError(f"{level_name} 没有可生成 cascade 的验证样本")
    write_model_report(
        output_dir,
        class_names,
        paths,
        labels,
        predictions,
        sample_name="val_eval_results.csv",
        extra_summary={
            "sources": [{"kind": "cascade", "level_name": level_name, "sample_count": len(paths)}],
        },
        metric_class_ids=sorted(set(labels)),
    )


def _collect_level_frames(
    context: EvaluationContext,
    level_name: str,
    result_dir_name: str,
    sample_csv_name: str,
    device: Any,
    predictor_cache: dict[Path, tuple[Any, Any]],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    """收集一个层级下所有模型槽位或单子类继承的结果行。"""
    level_dir = context.experiment_dir / level_name
    parent_to_children = (
        (context.meta.get("parent_to_children") or {}).get(level_name) or {}
    )
    global_entry = (context.meta.get("level_models") or {}).get(level_name)
    if has_model_path(global_entry):
        frame = _read_slot_results(
            context,
            level_name,
            level_dir,
            result_dir_name,
            sample_csv_name,
        )
        if frame is None:
            raise FileNotFoundError(
                f"{level_name} 缺少 {result_dir_name}，请先评估全局模型"
            )
        return [frame], [
            {
                "kind": "model",
                "run_dir": relpath(level_dir, context.experiment_dir),
                "parent_id": None,
                "sample_count": int(len(frame)),
            }
        ]

    parent_level = context.dataset_index.get_parent_level(level_name)
    if parent_level is None:
        raise FileNotFoundError(f"{level_name} 没有父层且缺少全局模型")
    if not parent_to_children:
        raise RuntimeError(f"{level_name} 没有可汇总的父类模型")
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for parent_text, child_ids in sorted(
        parent_to_children.items(), key=lambda item: int(item[0])
    ):
        parent_id = int(parent_text)
        if len(child_ids) > 1:
            slot = level_dir / f"{level_name}_{parent_id}"
            frame = _read_slot_results(
                context,
                level_name,
                slot,
                result_dir_name,
                sample_csv_name,
            )
            if frame is None:
                if result_dir_name == TEST_RESULT_DIR_NAME:
                    sources.append(
                        {
                            "kind": "model",
                            "run_dir": relpath(slot, context.experiment_dir),
                            "parent_id": parent_id,
                            "sample_count": 0,
                        }
                    )
                    continue
                raise FileNotFoundError(
                    f"{level_name} 缺少父类 {parent_id} 的 {result_dir_name}，"
                    "请先评估该模型"
                )
            frames.append(frame)
            sources.append(
                {
                    "kind": "model",
                    "run_dir": relpath(slot, context.experiment_dir),
                    "parent_id": parent_id,
                    "sample_count": int(len(frame)),
                }
            )
            continue
        frame, source = _inherit_single_child(
            context,
            level_name,
            parent_level,
            parent_id,
            int(child_ids[0]),
            result_dir_name,
            sample_csv_name,
            device,
            predictor_cache,
        )
        if not frame.empty:
            frames.append(frame)
        sources.append(source)
    if not frames:
        raise RuntimeError(f"{level_name} 没有可汇总的 {result_dir_name} 结果")
    return frames, sources


def _inherit_single_child(
    context: EvaluationContext,
    level_name: str,
    parent_level: str,
    parent_id: int,
    child_id: int,
    result_dir_name: str,
    sample_csv_name: str,
    device: Any,
    predictor_cache: dict[Path, tuple[Any, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """继承父层模型的预测；误判样本继续由误判父类的子模型级联预测。"""
    parent_level_dir = context.experiment_dir / parent_level
    matched_frames: list[pd.DataFrame] = []
    parent_slots: list[Path] = []
    parent_result_found = False
    for parent_slot in find_model_slot_dirs(parent_level_dir):
        frame = _read_slot_results(
            context,
            parent_level,
            parent_slot,
            result_dir_name,
            sample_csv_name,
        )
        if frame is None:
            continue
        parent_result_found = True
        matched = frame[frame["label_true"] == parent_id]
        if matched.empty:
            continue
        matched_frames.append(matched)
        parent_slots.append(parent_slot)
    if not parent_result_found:
        raise FileNotFoundError(
            f"层级汇总需要先评估 {parent_level} 的模型（{result_dir_name}）"
        )
    if len(matched_frames) > 1:
        raise RuntimeError(
            f"{parent_level} 有多个槽位包含父类 {parent_id} 的样本，无法继承"
        )
    if not matched_frames:
        empty = pd.DataFrame({"path": [], "label_true": [], "label_pred": []})
        return empty, {
            "kind": "inherited_single_child",
            "parent_level": parent_level,
            "parent_run_dir": None,
            "parent_id": parent_id,
            "child_id": child_id,
            "sample_count": 0,
        }

    parent_frame = matched_frames[0]
    parent_slot = parent_slots[0]
    target_predictions: list[int] = []
    for parent_prediction, sample_path in zip(
        parent_frame["label_pred"].astype(int),
        parent_frame["path"].astype(str),
    ):
        if int(parent_prediction) == parent_id:
            target_predictions.append(child_id)
            continue
        target_predictions.append(
            _predict_routed_child(
                context,
                level_name,
                int(parent_prediction),
                sample_path,
                device,
                predictor_cache,
            )
        )
    frame = pd.DataFrame(
        {
            "path": parent_frame["path"].values,
            "label_true": [child_id] * len(parent_frame),
            "label_pred": target_predictions,
        }
    )
    return frame, {
        "kind": "inherited_single_child",
        "parent_level": parent_level,
        "parent_run_dir": relpath(parent_slot, context.experiment_dir),
        "parent_id": parent_id,
        "child_id": child_id,
        "sample_count": int(len(frame)),
    }


def _predict_routed_child(
    context: EvaluationContext,
    level_name: str,
    parent_id: int,
    sample_path: str,
    device: Any,
    predictor_cache: dict[Path, tuple[Any, Any]],
) -> int:
    """按父层误判类别选择子模型，并对同一光谱继续级联预测。"""
    children = (
        ((context.meta.get("parent_to_children") or {}).get(level_name) or {})
        .get(str(parent_id))
        or []
    )
    if not children:
        raise ValueError(f"{level_name} 的父类 {parent_id} 没有子类，无法继续级联")
    if len(children) == 1:
        return int(children[0])
    slot = context.experiment_dir / level_name / f"{level_name}_{parent_id}"
    if not (slot / MODEL_FILE_NAME).is_file():
        raise FileNotFoundError(
            f"层级汇总的继承样本需要 {level_name} 父类 {parent_id} 的子模型：{slot}"
        )
    cached = predictor_cache.get(slot)
    if cached is None:
        predictor = load_predictor(slot, device, level_name)
        preprocessor = build_inference_preprocessor(
            context.input_spec,
            predictor.device,
        )
        predictor_cache[slot] = (predictor, preprocessor)
    predictor, preprocessor = predictor_cache[slot]
    inputs = preprocessor.preprocess_intensity(load_arc_intensity(sample_path))
    return int(predictor.predict_tensor(inputs, top_k=1)[0].class_id)


def _read_slot_results(
    context: EvaluationContext,
    level_name: str,
    slot_dir: Path,
    result_dir_name: str,
    sample_csv_name: str,
) -> pd.DataFrame | None:
    """读取一个模型槽位的结果 CSV，并把局部标签映射到层级全局类别。"""
    csv_path = slot_dir / result_dir_name / sample_csv_name
    if not csv_path.is_file():
        return None
    entry, parent_id = _resolve_slot_entry(context, level_name, slot_dir)
    class_ids = _slot_class_ids(context, level_name, parent_id, entry)
    local_to_global = {
        local_id: int(global_id) for local_id, global_id in enumerate(class_ids)
    }
    frame = pd.read_csv(csv_path)
    frame["label_true"] = frame["label_true"].map(local_to_global)
    frame["label_pred"] = frame["label_pred"].map(local_to_global)
    if frame["label_true"].isna().any() or frame["label_pred"].isna().any():
        raise ValueError(f"结果 CSV 标签超出模型类别范围：{csv_path}")
    frame["label_true"] = frame["label_true"].astype(int)
    frame["label_pred"] = frame["label_pred"].astype(int)
    return frame[["path", "label_true", "label_pred"]]


def _resolve_slot_entry(
    context: EvaluationContext,
    level_name: str,
    slot_dir: Path,
) -> tuple[Any, int | None]:
    """按槽位目录定位全局或父类子模型条目。"""
    level_dir = context.experiment_dir / level_name
    if slot_dir.resolve() == level_dir.resolve():
        entry = (context.meta.get("level_models") or {}).get(level_name)
        parent_id = None
    else:
        prefix = f"{level_name}_"
        if not slot_dir.name.startswith(prefix):
            raise ValueError(f"无法识别的模型槽位目录：{slot_dir}")
        parent_id = int(slot_dir.name[len(prefix) :])
        entry = (
            ((context.meta.get("parent_models") or {}).get(level_name) or {})
            .get(str(parent_id))
        )
    if not has_model_path(entry):
        raise FileNotFoundError(f"{level_name} 的槽位未记录模型：{slot_dir}")
    run_dir = resolve_entry_path(context.experiment_dir, entry, "run_dir")
    if run_dir != slot_dir.resolve():
        raise ValueError(f"槽位目录与层级元数据不一致：{slot_dir}")
    return entry, parent_id


def _slot_class_ids(
    context: EvaluationContext,
    level_name: str,
    parent_id: int | None,
    entry: Any,
) -> list[int]:
    """返回槽位输出类别在目标层级的全局索引顺序。"""
    if parent_id is None:
        return list(range(context.dataset_index.num_classes_by_level[level_name]))
    return [int(item) for item in entry.get("child_ids") or []]


__all__ = [
    "LEVEL_SUMMARY_DIR_NAME",
    "assert_multi_model_level",
    "summarize_level_results",
]

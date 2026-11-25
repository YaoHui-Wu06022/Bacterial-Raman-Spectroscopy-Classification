"""前置流水线唯一的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ramanv2.common.file_ops import write_json
from ramanv2.data.naming import BEAD_FOLDER_NAME
from ramanv2.data.dataset_mapping import MAPPING_FILE_NAME, ensure_dataset_mapping
from ramanv2.data.profiles import resolve_selection
from ramanv2.pipeline.state import pipeline_state_path_for_spec


DEFAULT_DATASET_ID = None


def configure_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("clean-status", help="查看前置流水线状态")
    status.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="数据集标识，默认使用默认数据集")
    status.set_defaults(run_command=run_command)

    affine = commands.add_parser("affine", help="分析并发布仿射校准日期")
    affine.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="数据集标识，默认使用默认数据集")
    affine.add_argument("--source-root")
    affine.add_argument("--output-root")
    affine.add_argument("--date", dest="date_names", action="append")
    affine.set_defaults(run_command=run_command)

    init = commands.add_parser("init", help="从 shift_data 生成全量 init")
    init.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="数据集标识，默认使用默认数据集")
    init.add_argument("--shift-root")
    init.add_argument("--init-root")
    init.add_argument("--test-folder", dest="test_folders", action="append", default=[])
    init.set_defaults(run_command=run_command)

    plots = commands.add_parser("plot-prefix", help="生成指定 init 目录的 prefix-plot")
    plots.add_argument("--input-root", required=True)
    plots.add_argument("--output-root", required=True)
    plots.add_argument("--profile", default=None, help="可选 profile；提供时按该 profile 层级扫描目录")
    plots.add_argument("--force", action="store_true", dest="force_enable")
    plots.set_defaults(run_command=run_command)

    rebuild = commands.add_parser("rebuild-shift", help="重建指定日期的 shift_data")
    rebuild.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="数据集标识，默认使用默认数据集")
    rebuild.add_argument("--date", dest="date_names", action="append", required=True)
    rebuild.set_defaults(run_command=run_command)

    records = commands.add_parser("rebuild-records", help="按当前磁盘重建走向树与页面状态")
    records.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="数据集标识，默认使用默认数据集")
    records.set_defaults(run_command=run_command)

    db_check = commands.add_parser("db-check", help="检查 SQLite 元数据与磁盘指纹差异")
    db_check.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="数据集标识，默认使用默认数据集")
    db_check.add_argument("--refresh", action="store_true", dest="refresh_fingerprints")
    db_check.set_defaults(run_command=run_command)

    upgrade = commands.add_parser("upgrade-db", help="备份并升级数据集流水线数据库到 v2")
    upgrade.add_argument("--dataset", required=True, help="ASCII 数据集标识")
    upgrade.set_defaults(run_command=run_command)

    ingest = commands.add_parser("ingest-folder", help="清洗并入库一个文件夹")
    ingest.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="数据集标识，默认使用默认数据集")
    ingest.add_argument("--source", help="源文件夹；非 ASCII 路径优先用 --manifest")
    ingest.add_argument("--date", help="八位日期，例如 20240101")
    ingest.add_argument("--prefix", help="入库文件夹前缀，例如 SMA01")
    ingest.add_argument("--bead", action="store_true", dest="bead_enable", help="小球批次：只去文件头，不做宇宙射线去除")
    ingest.add_argument("--manifest", help="UTF-8 JSON 清单：source_dir/date/prefix")
    ingest.add_argument("--apply", action="store_true", dest="apply_enable", help="实际生成 cosmic_data；默认只预览")
    ingest.add_argument("--json", action="store_true", dest="json_enable", help="输出完整 JSON 报告")
    ingest.set_defaults(run_command=run_command)


def _dataset_mapping_for_spec(spec):
    """按 label_source 读取或生成当前数据集的通用前缀映射。"""
    return ensure_dataset_mapping(
        spec.label_source,
        spec.directory,
        spec.metadata_dir / MAPPING_FILE_NAME,
        raw_dir_name=spec.raw_dir.name,
    )


def run_command(args: argparse.Namespace) -> int:
    selection = resolve_selection(
        getattr(args, "dataset", DEFAULT_DATASET_ID),
        getattr(args, "profile", None),
    )
    spec = selection.spec
    if args.command == "ingest-folder":
        from ramanv2.pipeline.folder_ingest import (
            build_folder_ingest_request,
            format_ingest_summary,
            load_folder_ingest_manifest,
            run_folder_ingest,
        )

        source = args.source
        date_name = args.date
        prefix = args.prefix
        bead = bool(getattr(args, "bead_enable", False))
        if args.manifest:
            manifest = load_folder_ingest_manifest(args.manifest)
            source = source or manifest.get("source_dir") or manifest.get("source")
            date_name = date_name or manifest.get("date")
            prefix = prefix or manifest.get("prefix")
            bead = bead or bool(manifest.get("bead"))
        if not source or not date_name or (not bead and not prefix):
            raise ValueError("ingest-folder 需要 --source/--date/--prefix，或使用 --manifest")
        report = run_folder_ingest(build_folder_ingest_request(
            spec,
            source,
            str(date_name),
            prefix,
            apply=bool(args.apply_enable),
            bead=bead,
        ))
        if args.json_enable:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(format_ingest_summary(report))
        return 0
    if args.command in {"affine", "rebuild-shift", "rebuild-records"} and not spec.cosmic_dir.is_dir():
        print("cosmic_data 不存在：请先在「宇宙射线去除」页从 raw_data 生成。")
        return 1
    if args.command == "upgrade-db":
        from ramanv2.pipeline.db import upgrade_database

        report = upgrade_database(pipeline_state_path_for_spec(spec))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "db-check":
        from ramanv2.pipeline.db import check_database, refresh_fingerprints

        metadata_dir = spec.metadata_dir
        cosmic_root = spec.cosmic_dir
        if args.refresh_fingerprints:
            refresh_fingerprints(metadata_dir, cosmic_root)
        print(json.dumps(check_database(metadata_dir, cosmic_root), ensure_ascii=False, indent=2))
        return 0

    from ramanv2.pipeline.state import write_pipeline_state
    from ramanv2.pipeline.lineage import load_pipeline_progress

    state_path = pipeline_state_path_for_spec(spec)
    cosmic_root = spec.cosmic_dir
    state = load_pipeline_progress(
        cosmic_root, state_path, initialize_enable=args.command != "init",
    )
    if args.command == "clean-status":
        print(json.dumps(state.to_payload(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "affine":
        from ramanv2.pipeline.calibration import record_affine_reports, run_affine_calibration, run_identity_affine

        dates = args.date_names or sorted(state.cleaning.completed_date_names)
        source_root = args.source_root or spec.cosmic_dir
        output_root = args.output_root or spec.shift_dir
        from ramanv2.data.arc_data import has_arc_data

        if not dates and spec.cosmic_dir.is_dir():
            from ramanv2.pipeline.cleaning import scan_date_names

            dates = scan_date_names(spec.cosmic_dir)
        bead_dates = [
            date_name for date_name in dates
            if has_arc_data(Path(source_root) / date_name / BEAD_FOLDER_NAME, recursive=True)
        ]
        flat_dates = [date_name for date_name in dates if date_name not in set(bead_dates)]
        reports = run_affine_calibration(source_root, bead_dates, output_root)
        reports += run_identity_affine(source_root, flat_dates, output_root)
        record_affine_reports(state, state_path, reports, pop_latest=True)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "init":
        from ramanv2.pipeline.supplement import run_full_init

        report = run_full_init(
            spec,
            state,
            state_path,
            _dataset_mapping_for_spec(spec),
            shift_root=args.shift_root,
            init_root=args.init_root,
            selected_test_folders=args.test_folders,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "rebuild-shift":
        from ramanv2.pipeline.calibration import rebuild_shift_data
        from ramanv2.pipeline.state import invalidate_downstream

        reports = rebuild_shift_data(
            spec.cosmic_dir,
            spec.shift_dir,
            args.date_names,
        )
        for report in reports:
            date_name = str(report["date"])
            if report.get("status") == "rebuilt":
                state.affine_completed_dates.add(date_name)
                invalidate_downstream(state, [date_name])
        report_path = spec.metadata_dir / "shift_rebuild_report.json"
        write_json(report_path, reports)
        write_pipeline_state(state_path, state)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "rebuild-records":
        from ramanv2.pipeline.db import refresh_fingerprints
        from ramanv2.pipeline.reconcile import rebuild_pipeline_records

        catalog = _dataset_mapping_for_spec(spec)
        summary = rebuild_pipeline_records(
            cosmic_root,
            spec.shift_dir,
            catalog,
            state_path,
            dataset_id=spec.dataset_id,
        )
        summary["fingerprints"] = refresh_fingerprints(
            spec.metadata_dir,
            cosmic_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "plot-prefix":
        from ramanv2.data.plots.prefix import build_prefix_plots

        hierarchy = (
            resolve_selection(profile_id=args.profile).hierarchy
            if getattr(args, "profile", None)
            else None
        )
        outputs = build_prefix_plots(
            args.input_root,
            args.output_root,
            args.force_enable,
            hierarchy=hierarchy,
        )
        print(f"figures={len(outputs)}")
        return 0
    raise RuntimeError(f"未知 pipeline 命令：{args.command}")


__all__ = ["configure_parser", "run_command"]

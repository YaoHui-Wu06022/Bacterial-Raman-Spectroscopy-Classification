"""前置流水线唯一的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ramanv2.common.file_ops import write_json
from ramanv2.common.naming import BEAD_FOLDER_NAME
from ramanv2.data.datasets import annotate_dataset_build
from ramanv2.data.dataset_mapping import ensure_dataset_mapping
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

    acquire = commands.add_parser("acquire", help="从全量 init 生成 profile 子集")
    acquire.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="数据集标识，默认使用默认数据集")
    acquire.add_argument("--init-root")
    acquire.add_argument("--profile", required=True, help="数据集内 profile 标识")
    acquire.set_defaults(run_command=run_command)

    plots = commands.add_parser("plot-prefix", help="生成指定 init 目录的 prefix-plot")
    plots.add_argument("--input-root", required=True)
    plots.add_argument("--output-root", required=True)
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


def _dataset_mapping_for_spec(spec):
    """按数据集规格读取 profile 名单；扁平数据集没有 profile 名单。"""
    if spec.is_flat:
        return None
    return ensure_dataset_mapping(
        spec.workbook_path,
        spec.metadata_dir / "profile_catalog.json",
        spec.workbook_columns,
        spec.profile_rules,
    )


def run_command(args: argparse.Namespace) -> int:
    selection = resolve_selection(
        getattr(args, "dataset", DEFAULT_DATASET_ID),
        getattr(args, "profile", None),
    )
    spec = selection.spec
    if args.command == "db-check":
        from ramanv2.pipeline.db import check_database, refresh_fingerprints

        metadata_dir = spec.metadata_dir
        cosmic_root = spec.cosmic_dir
        if args.refresh_fingerprints:
            refresh_fingerprints(metadata_dir, cosmic_root)
        print(json.dumps(check_database(metadata_dir, cosmic_root), ensure_ascii=False, indent=2))
        return 0

    from ramanv2.pipeline.state import load_pipeline_state, write_pipeline_state
    from ramanv2.records.lineage import initialize_folder_lineage, sync_pipeline_state_from_lineage

    state_path = pipeline_state_path_for_spec(spec)
    state = load_pipeline_state(state_path)
    cosmic_root = spec.cosmic_dir
    if cosmic_root.is_dir():
        initialize_folder_lineage(cosmic_root)
        sync_pipeline_state_from_lineage(cosmic_root, state)
    if args.command == "clean-status":
        print(json.dumps(state.to_payload(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "affine":
        from ramanv2.pipeline.calibration import run_affine_calibration, run_identity_affine
        from ramanv2.pipeline.state import invalidate_downstream

        dates = args.date_names or sorted(state.cleaning.completed_date_names)
        source_root = args.source_root or spec.cosmic_dir
        output_root = args.output_root or spec.shift_dir
        from ramanv2.common.arc_data import has_arc_data

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
        for report in reports:
            date_name = str(report["date"])
            if report.get("status") == "manual_approved":
                state.affine_completed_dates.add(date_name)
                if not report.get("skipped"):
                    invalidate_downstream(state, [date_name])
            elif not report.get("skipped"):
                state.affine_completed_dates.discard(date_name)
        state.dataset_builds.pop("latest", None)
        write_pipeline_state(state_path, state)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "init":
        from ramanv2.pipeline.supplement import (
            SupplementPlan,
            build_full_init_with_supplements,
            scan_supplement_plan,
            validate_supplement_plan,
        )

        shift_root = args.shift_root or spec.shift_dir
        init_root = args.init_root or spec.init_dir
        catalog = _dataset_mapping_for_spec(spec)
        if spec.is_flat:
            plan = SupplementPlan(groups=(), errors=(), unbacked_init_folders=())
        else:
            plan = scan_supplement_plan(spec.supplement_path, init_root, catalog)
            errors = validate_supplement_plan(plan)
            if errors:
                raise ValueError("补充数据校验失败：" + "；".join(errors))
        report = build_full_init_with_supplements(
            shift_root,
            init_root,
            catalog,
            sorted(state.affine_completed_dates),
            plan,
            test_root=spec.init_test_dir,
            selected_test_folders=args.test_folders,
            hierarchy=spec.hierarchy,
        )
        state.dataset_builds["full_init"] = annotate_dataset_build(report, spec)
        write_pipeline_state(state_path, state)
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
    if args.command == "acquire":
        from ramanv2.data.builders.init import build_profile_dataset

        catalog = _dataset_mapping_for_spec(spec)
        profile = selection.profile
        profile_root = selection.dataset_dir / profile.root_init
        report = build_profile_dataset(args.init_root or spec.init_dir, profile_root, profile.profile_id, catalog)
        profiles = state.dataset_builds.setdefault("profiles", {})
        if not isinstance(profiles, dict):
            profiles = {}
            state.dataset_builds["profiles"] = profiles
        profiles[profile.profile_id] = report
        plot_reports = state.dataset_builds.get("prefix_plots")
        if isinstance(plot_reports, dict):
            plot_reports.pop(profile.profile_id, None)
        write_pipeline_state(state_path, state)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "plot-prefix":
        from ramanv2.data.plots.prefix import build_prefix_plots

        outputs = build_prefix_plots(args.input_root, args.output_root, args.force_enable)
        print(f"figures={len(outputs)}")
        return 0
    raise RuntimeError(f"未知 pipeline 命令：{args.command}")


__all__ = ["configure_parser", "run_command"]

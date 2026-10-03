"""前置流水线唯一的命令行入口。"""

from __future__ import annotations

import argparse
import json

from ramanv2.core.paths import PROJECT_ROOT
from ramanv2.data.profiles import PIPELINE_PROFILE_IDS
from ramanv2.pipeline.state import pipeline_metadata_dir, pipeline_state_path


def configure_parser(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("clean-status", help="查看前置流水线状态")
    status.set_defaults(run_command=run_command)

    affine = commands.add_parser("affine", help="分析并发布仿射校准日期")
    affine.add_argument("--source-root", default=str(PROJECT_ROOT / "dataset" / "cosmic_data"))
    affine.add_argument("--output-root", default=str(PROJECT_ROOT / "dataset" / "shift_data"))
    affine.add_argument("--date", dest="date_names", action="append")
    affine.set_defaults(run_command=run_command)

    audit = commands.add_parser("audit", help="审核并应用校正后文件夹异常谱")
    audit.add_argument("--shift-root", default=str(PROJECT_ROOT / "dataset" / "shift_data"))
    audit.add_argument("--date", dest="date_names", action="append")
    audit.set_defaults(run_command=run_command)

    init = commands.add_parser("init", help="从 shift_data 生成全量 init")
    init.add_argument("--shift-root", default=str(PROJECT_ROOT / "dataset" / "shift_data"))
    init.add_argument("--init-root", default=str(PROJECT_ROOT / "dataset" / "init"))
    init.add_argument("--cs-folder", dest="cs_folders", action="append", default=[])
    init.set_defaults(run_command=run_command)

    acquire = commands.add_parser("acquire", help="从全量 init 生成 profile 子集")
    acquire.add_argument("--init-root", default=str(PROJECT_ROOT / "dataset" / "init"))
    acquire.add_argument("--profile", choices=PIPELINE_PROFILE_IDS, default=PIPELINE_PROFILE_IDS[0])
    acquire.set_defaults(run_command=run_command)

    plots = commands.add_parser("plot-prefix", help="生成指定 init 目录的 prefix-plot")
    plots.add_argument("--input-root", required=True)
    plots.add_argument("--output-root", required=True)
    plots.add_argument("--force", action="store_true", dest="force_enable")
    plots.set_defaults(run_command=run_command)

    rebuild = commands.add_parser("rebuild-shift", help="重建指定日期的 shift_data")
    rebuild.add_argument("--date", dest="date_names", action="append", required=True)
    rebuild.set_defaults(run_command=run_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raman 前置数据流水线")
    configure_parser(parser)
    return parser


def _pending_audit_dates(state) -> list[str]:
    """返回默认 CLI 审核范围，排除已完成日期。"""
    return sorted(state.affine_completed_dates - state.audit_completed_dates)


def run_command(args: argparse.Namespace) -> int:
    from ramanv2.pipeline.state import load_pipeline_state, save_pipeline_state
    from ramanv2.pipeline.lineage import initialize_folder_lineage, sync_pipeline_state_from_lineage

    state_path = pipeline_state_path(PROJECT_ROOT)
    state = load_pipeline_state(state_path)
    cosmic_root = PROJECT_ROOT / "dataset" / "cosmic_data"
    if cosmic_root.is_dir():
        initialize_folder_lineage(cosmic_root)
        sync_pipeline_state_from_lineage(cosmic_root, state)
    if args.command == "clean-status":
        print(json.dumps(state.to_payload(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "affine":
        from ramanv2.pipeline.calibration import run_affine_calibration

        dates = args.date_names or sorted(state.cleaning.completed_date_names)
        reports = run_affine_calibration(args.source_root, dates, args.output_root)
        for report in reports:
            date_name = str(report["date"])
            if report.get("status") == "manual_approved":
                state.affine_completed_dates.add(date_name)
                if not report.get("skipped"):
                    state.audit_completed_dates.discard(date_name)
            elif not report.get("skipped"):
                state.affine_completed_dates.discard(date_name)
        state.dataset_builds.pop("latest", None)
        save_pipeline_state(state_path, state)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit":
        from ramanv2.pipeline.audit import run_shift_folder_audit

        dates = args.date_names or _pending_audit_dates(state)
        reports = run_shift_folder_audit(args.shift_root, dates, state, state_path)
        state.dataset_builds.pop("latest", None)
        save_pipeline_state(state_path, state)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "init":
        from ramanv2.data.builders.init import build_full_init
        from ramanv2.data.catalog import ensure_dataset_mapping

        workbook = PROJECT_ROOT / "dataset" / "病原菌分类与规范简称.xlsx"
        mapping_file = pipeline_metadata_dir(PROJECT_ROOT) / "profile_catalog.json"
        catalog = ensure_dataset_mapping(workbook, mapping_file)
        report = build_full_init(
            args.shift_root,
            args.init_root,
            catalog,
            sorted(state.audit_completed_dates & state.affine_completed_dates),
            test_root=PROJECT_ROOT / "dataset" / "CSdata",
            selected_cs_folders=args.cs_folders,
        )
        state.dataset_builds["full_init"] = report
        state.dataset_builds.pop("profiles", None)
        state.dataset_builds.pop("prefix_plots", None)
        save_pipeline_state(state_path, state)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "rebuild-shift":
        from ramanv2.pipeline.calibration import rebuild_shift_dates

        reports = rebuild_shift_dates(
            PROJECT_ROOT / "dataset" / "cosmic_data",
            PROJECT_ROOT / "dataset" / "shift_data",
            args.date_names,
        )
        for report in reports:
            date_name = str(report["date"])
            if report.get("status") == "manual_approved":
                state.affine_completed_dates.add(date_name)
            else:
                state.affine_completed_dates.discard(date_name)
            state.audit_completed_dates.discard(date_name)
        state.dataset_builds.pop("full_init", None)
        state.dataset_builds.pop("profiles", None)
        state.dataset_builds.pop("prefix_plots", None)
        report_path = pipeline_metadata_dir(PROJECT_ROOT) / "shift_rebuild_report.json"
        report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        save_pipeline_state(state_path, state)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "acquire":
        from ramanv2.data.builders.init import build_profile_dataset
        from ramanv2.data.catalog import ensure_dataset_mapping
        from ramanv2.data.profiles import get_dataset_dir, get_profile

        workbook = PROJECT_ROOT / "dataset" / "病原菌分类与规范简称.xlsx"
        mapping_file = pipeline_metadata_dir(PROJECT_ROOT) / "profile_catalog.json"
        catalog = ensure_dataset_mapping(workbook, mapping_file)
        profile = get_profile(args.profile)
        profile_root = get_dataset_dir(profile, PROJECT_ROOT) / profile.root_init
        report = build_profile_dataset(args.init_root, profile_root, args.profile, catalog)
        profiles = state.dataset_builds.setdefault("profiles", {})
        if not isinstance(profiles, dict):
            profiles = {}
            state.dataset_builds["profiles"] = profiles
        profiles[args.profile] = report
        plot_reports = state.dataset_builds.get("prefix_plots")
        if isinstance(plot_reports, dict):
            plot_reports.pop(args.profile, None)
        save_pipeline_state(state_path, state)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "plot-prefix":
        from ramanv2.data.plots.prefix import build_prefix_plots

        outputs = build_prefix_plots(args.input_root, args.output_root, args.force_enable)
        print(f"figures={len(outputs)}")
        return 0
    raise RuntimeError(f"未知 pipeline 命令：{args.command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run_command(args)


__all__ = ["build_parser", "configure_parser", "main", "run_command"]

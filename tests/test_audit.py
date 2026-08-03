import csv
import shutil
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import raman_temp.audit.plots as target_plots
import raman_temp.audit.similarity as target_similarity
import raman_temp.audit.workflow as target_workflow

from raman.audit import workflow as reference_workflow
from raman.audit import test_pool as reference_test_pool
from raman.audit.common import preprocess_spectrum_for_audit as reference_preprocess_audit_spectrum
from raman.data.profiles import get_profile as get_reference_profile
from raman.tool.dataset import resolve_dataset
from raman_temp.audit.config import DEFAULT_AUDIT_CONFIG
from raman_temp.audit.cli import build_parser as build_audit_parser
from raman_temp.audit.preprocess import preprocess_audit_spectrum
from raman_temp.audit.preprocess import AuditSpectrum
from raman_temp.audit.plots import PlotPaths, plot_prefix_dataset, plot_shift_folder
from raman_temp.audit.raw_quality import run_stage1, score_raw_record
from raman_temp.audit.records import RawRecord
from raman_temp.audit.records import SimilarityRecord
from raman_temp.audit.similarity import run_stage2, run_stage3, score_similarity_group
from raman_temp.audit.reports import read_delta, write_delta
from raman_temp.audit.shift import ManualShiftTarget, apply_folder_shift, apply_manual_shift, build_folder_curve, find_peak_anchor
from raman_temp.audit.test_sync import build_candidate_commit_rows, build_work_pool, commit_candidate_rows, sync_work_test_init
from raman_temp.data.test_transfer import TestTransferConfig, build_test_transfer
from raman_temp.data.profiles import get_profile


def test_raw_quality_metrics_match_reference_on_random_raw_samples():
    source_root = "dataset/alldata/raw"
    paths = sorted(Path(source_root).rglob("*.arc_data"))
    assert paths
    indices = np.random.default_rng(20260803).choice(len(paths), size=min(3, len(paths)), replace=False)

    for index in indices:
        path = paths[int(index)]
        reference = reference_workflow.RawRecord(path, "", "", "", "")
        reference_workflow._raw_metrics(reference)
        target = RawRecord(path=path, rel_path="", genus="", folder="", prefix="")
        score_raw_record(target, DEFAULT_AUDIT_CONFIG)

        assert target.state == reference.state
        assert target.reasons == reference.reasons
        assert target.points == reference.points
        assert target.malformed_lines == reference.malformed_lines
        assert target.longest_flat_points == reference.longest_flat_points
        assert target.saturation_points == reference.saturation_points
        np.testing.assert_allclose(target.coverage, reference.coverage, equal_nan=True)
        np.testing.assert_allclose(target.noise_ratio, reference.noise_ratio, equal_nan=True)


def test_audit_config_preserves_shift_special_cases_without_dataset_lookup():
    config = DEFAULT_AUDIT_CONFIG
    assert config.fungal_genera == {"Candida"}
    assert config.shift_large_move_folders == reference_workflow.SHIFT_LARGE_MOVE_FOLDERS
    assert config.get_fixed_shift("Klebsiella", "KAE03") == reference_workflow.SHIFT_FIXED_TOTALS[("Klebsiella", "KAE03")]
    assert config.get_fixed_shift("Klebsiella", "KAE01") is None


def test_audit_preprocess_matches_reference_on_random_raw_sample():
    paths = sorted(Path("dataset/alldata/raw").rglob("*.arc_data"))
    assert paths
    path = paths[np.random.default_rng(20260803).integers(len(paths))]
    reference = reference_preprocess_audit_spectrum(path, get_reference_profile("alldata"))
    target = preprocess_audit_spectrum(path, get_profile("alldata"))

    assert target.skip_reason == reference["skip_reason"]
    np.testing.assert_allclose(target.wavenumbers, reference["wn"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(target.intensities, reference["sp"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(target.normalized, reference["z"], rtol=1e-6, atol=1e-6)


def test_similarity_metrics_and_candidate_match_reference():
    random_source = np.random.default_rng(20260803)
    base = np.sin(np.linspace(0.0, 6.0, 128))
    spectra = [base + random_source.normal(0.0, 0.01, base.size) for _ in range(9)]
    spectra[-1] = -base + random_source.normal(0.0, 0.01, base.size)

    reference_records = []
    target_records = []
    for index, spectrum in enumerate(spectra):
        path = Path(f"sample_{index}.arc_data")
        reference_raw = reference_workflow.RawRecord(path, f"G/AA01/{path.name}", "G", "AA01", "AA")
        target_raw = RawRecord(path, f"G/AA01/{path.name}", "G", "AA01", "AA")
        reference_records.append(reference_workflow.SimilarityRecord(raw=reference_raw, z=spectrum.astype(np.float32)))
        target_records.append(SimilarityRecord(raw=target_raw, spectrum=spectrum.astype(np.float32)))

    reference_workflow._score_similarity_group(reference_records)
    score_similarity_group(target_records)

    for reference, target in zip(reference_records, target_records):
        assert target.reference_count == reference.ref_count
        assert target.neighbor_count == reference.neighbor_count
        np.testing.assert_allclose(target.neighbor_corr, reference.neighbor_corr, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(target.rmse, reference.rmse, rtol=1e-6, atol=1e-6)
        assert (target.state == "candidate") == (reference.state == "candidate")


def test_stage2_and_stage3_share_scores_without_pair_fields(tmp_path, monkeypatch):
    input_dir = tmp_path / "work" / "pool" / "init"
    for folder in ("AA01", "AA02"):
        for index in range(9):
            path = input_dir / "Genus" / folder / f"sample_{index}.arc_data"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("600\t1\n", encoding="utf-8")

    axis = np.linspace(600.0, 1800.0, 32, dtype=np.float32)

    def build_fake_spectrum(path, profile, config, reference_wavenumbers, label):
        index = int(path.stem.rsplit("_", 1)[1])
        values = np.sin(axis / 100.0) + index * 0.001
        return AuditSpectrum(axis, values, values)

    monkeypatch.setattr(target_similarity, "preprocess_audit_spectrum", build_fake_spectrum)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    work_dir = tmp_path / "work"
    profile = get_profile("alldata")
    stage2 = run_stage2(input_dir, work_dir, run_dir, profile)
    stage3 = run_stage3(input_dir, work_dir, run_dir, profile)

    assert stage2.record_count == 18
    assert stage3.record_count == 18
    header = (stage3.stage_dir / "stage3_scores.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "pair_corr" not in header
    assert "local_area" not in header


def test_shift_curve_and_file_write_match_reference(tmp_path):
    folders = sorted(
        path.parent
        for path in Path("dataset/alldata/init").rglob("*.arc_data")
    )
    assert folders
    source_dir = folders[np.random.default_rng(20260803).integers(len(folders))]
    reference_curve = reference_workflow._folder_curve(source_dir)
    target_curve = build_folder_curve(source_dir, DEFAULT_AUDIT_CONFIG)
    if reference_curve is None:
        assert target_curve is None
    else:
        np.testing.assert_allclose(target_curve, reference_curve, rtol=1e-6, atol=1e-6)
        reference_anchor = reference_workflow._anchor_peak(reference_curve, reference_workflow.DEFAULT_PIPELINE_CONFIG.build_wn_ref())
        target_axis = np.linspace(
            DEFAULT_AUDIT_CONFIG.input.cut_min,
            DEFAULT_AUDIT_CONFIG.input.cut_max,
            DEFAULT_AUDIT_CONFIG.input.target_points,
            dtype=np.float32,
        )
        target_anchor = find_peak_anchor(target_curve, target_axis, DEFAULT_AUDIT_CONFIG)
        if reference_anchor is None:
            assert target_anchor is None
        else:
            assert target_anchor is not None
            np.testing.assert_allclose(target_anchor, reference_anchor, rtol=1e-6, atol=1e-6)

    source_path = next(source_dir.glob("*.arc_data"))
    reference_dir = tmp_path / "reference"
    target_dir = tmp_path / "target"
    reference_dir.mkdir()
    target_dir.mkdir()
    reference_path = reference_dir / source_path.name
    target_path = target_dir / source_path.name
    shutil.copy2(source_path, reference_path)
    shutil.copy2(source_path, target_path)
    reference_workflow._shift_file(reference_path, 1.2)
    assert apply_folder_shift(target_dir, 1.2) == 1
    assert target_path.read_bytes() == reference_path.read_bytes()


def test_manual_shift_syncs_matching_delta_and_rejects_mismatch(tmp_path):
    test_dir = tmp_path / "test" / "init" / "CS01KP"
    train_dir = tmp_path / "alldata" / "init" / "Klebsiella" / "KP02t"
    for folder in (test_dir, train_dir):
        folder.mkdir(parents=True)
        (folder / "sample.arc_data").write_text("600\t1\n601\t2\n", encoding="utf-8")
    test_delta = tmp_path / "test" / "delta.txt"
    train_delta = tmp_path / "alldata" / "delta.txt"
    test_target = ManualShiftTarget(test_dir, test_delta, ".", "CS01KP", "KP")
    train_target = ManualShiftTarget(train_dir, train_delta, "Klebsiella", "KP02t", "KP")

    result = apply_manual_shift(test_target, 0.5, train_target)
    assert result["cumulative_delta"] == 0.5
    assert read_delta(test_delta) == {(".", "CS01KP"): 0.5}
    assert read_delta(train_delta) == {("Klebsiella", "KP02t"): 0.5}

    write_delta(test_delta, {(".", "CS01KP"): ("KP", 0.5)})
    write_delta(train_delta, {("Klebsiella", "KP02t"): ("KP", 0.6)})
    before = (test_dir / "sample.arc_data").read_bytes()
    with pytest.raises(ValueError, match="对应 delta 不一致"):
        apply_manual_shift(test_target, 0.5, train_target)
    assert (test_dir / "sample.arc_data").read_bytes() == before


def test_candidate_commit_and_test_sync_use_source_mapping(tmp_path):
    work_dir = tmp_path / "run" / "work"
    main_candidate = work_dir / "delete" / "stage2" / "Genus" / "AA01" / "main.arc_data"
    test_candidate = work_dir / "delete" / "stage3" / "Genus" / "KP__CS01KP__audit" / "test.arc_data"
    for path in (main_candidate, test_candidate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("600\t1\n", encoding="utf-8")
    dataset_dir = tmp_path / "alldata"
    test_dir = tmp_path / "test"
    for path in (
        dataset_dir / "init" / "Genus" / "AA01" / "main.arc_data",
        test_dir / "init" / "CS01KP" / "test.arc_data",
        work_dir / "test" / "init" / "CS01KP" / "keep.arc_data",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("600\t1\n", encoding="utf-8")

    rows = build_candidate_commit_rows(work_dir)
    assert commit_candidate_rows(rows, dataset_dir, test_dir) == 2
    assert not (dataset_dir / "init" / "Genus" / "AA01" / "main.arc_data").exists()
    assert (dataset_dir / "delete" / "stage2" / "Genus" / "AA01" / "main.arc_data").is_file()
    assert not (test_dir / "init" / "CS01KP" / "test.arc_data").exists()
    assert sync_work_test_init(work_dir / "test" / "init", test_dir / "init", rows) == 1
    assert not (test_dir / "init" / "CS01KP" / "test.arc_data").exists()
    assert (test_dir / "init" / "CS01KP" / "keep.arc_data").is_file()


def test_clean_matches_reference_for_a_complete_species(tmp_path, monkeypatch):
    """用完整属目录和对应测试菌，对照候选提交、平移计划与最终 delta。"""
    source_dataset_dir = Path("dataset/alldata/raw/Klebsiella")
    source_test_dir = Path("dataset/测试菌/raw/CS01KP")
    assert len(list(source_dataset_dir.rglob("*.arc_data"))) == 424
    assert len(list(source_test_dir.glob("*.arc_data"))) == 47

    reference_root = tmp_path / "reference"
    target_root = tmp_path / "target"
    for root in (reference_root, target_root):
        shutil.copytree(source_dataset_dir, root / "alldata" / "init" / "Klebsiella")
        shutil.copytree(source_test_dir, root / "test" / "init" / "CS01KP")

    reference_dataset_dir = reference_root / "alldata"
    reference_test_dir = reference_root / "test"

    def resolve_reference_dataset(dataset_key, project_root=None, create=False):
        if dataset_key == "alldata":
            return get_reference_profile("alldata"), reference_dataset_dir
        if dataset_key == "test":
            return get_reference_profile("test"), reference_test_dir
        return resolve_dataset(dataset_key, project_root, create)

    monkeypatch.setattr(reference_workflow, "resolve_dataset", resolve_reference_dataset)
    monkeypatch.setattr(reference_test_pool, "resolve_dataset", resolve_reference_dataset)
    reference_test_pool.transfer_all()
    reference_workflow.run_stage1()
    reference_shift_dir = reference_workflow.run_data_driven_shift()
    reference_workflow.run_stage2()
    reference_workflow.run_stage3()

    target_dataset_dir = target_root / "alldata"
    target_test_dir = target_root / "test"
    monkeypatch.setattr(
        target_workflow,
        "get_dataset_dir",
        lambda profile: target_dataset_dir if profile.profile_id == "alldata" else target_test_dir,
    )
    run_dir = target_workflow.run_clean()

    def collect_delete_paths(dataset_dir):
        return {
            stage: sorted(
                path.relative_to(dataset_dir / "delete" / stage).as_posix()
                for path in (dataset_dir / "delete" / stage).rglob("*.arc_data")
            )
            if (dataset_dir / "delete" / stage).is_dir()
            else []
            for stage in ("stage1", "stage2", "stage3")
        }

    def read_shift_plan(plan_path):
        fields = (
            "genus",
            "folder",
            "prefix",
            "current_delta",
            "raw_from_zero_delta",
            "target_delta",
            "step_delta",
            "anchor_peak_cm",
            "anchor_model",
            "anchor_quality",
            "fit_bic_gain",
            "fit_separation_cm",
            "status",
            "residual_range",
            "total_shift_limit",
        )
        with plan_path.open("r", encoding="utf-8-sig", newline="") as file:
            return {
                (row["genus"], row["folder"]): tuple(row[field] for field in fields)
                for row in csv.DictReader(file)
            }

    def read_delta_log(log_path):
        if not log_path.is_file():
            return []
        with log_path.open("r", encoding="utf-8-sig", newline="") as file:
            return [
                {key: value for key, value in row.items() if key != "time"}
                for row in csv.DictReader(file, delimiter="\t")
            ]

    reference_dataset_delete = collect_delete_paths(reference_dataset_dir)
    reference_test_delete = collect_delete_paths(reference_test_dir)
    assert any(reference_dataset_delete.values()) or any(reference_test_delete.values())
    assert collect_delete_paths(target_dataset_dir) == reference_dataset_delete
    assert collect_delete_paths(target_test_dir) == reference_test_delete
    assert read_shift_plan(run_dir / "shift_plan.csv") == read_shift_plan(reference_shift_dir / "shift_plan.csv")
    assert read_delta(target_dataset_dir / "delta.txt") == read_delta(reference_dataset_dir / "delta.txt")
    assert read_delta(target_test_dir / "delta.txt") == read_delta(reference_test_dir / "delta.txt")
    assert read_delta_log(target_dataset_dir / "delta_log.txt") == read_delta_log(reference_dataset_dir / "delta_log.txt")
    assert read_delta_log(target_test_dir / "delta_log.txt") == read_delta_log(reference_test_dir / "delta_log.txt")
    assert not (target_dataset_dir / "fig_init").exists()
    assert (run_dir / "shift_plan.csv").is_file()
    assert (run_dir / "commit_plan.csv").is_file()
    assert (run_dir / "run.json").is_file()


def test_stage1_moves_only_the_work_pool_candidate(tmp_path):
    real_init = tmp_path / "real" / "init" / "Genus" / "AA01"
    real_init.mkdir(parents=True)
    source_path = real_init / "constant.arc_data"
    axis = np.linspace(600.0, 1800.0, 120)
    source_path.write_text(
        "".join(f"{value:.3f}\t1.000000\n" for value in axis),
        encoding="utf-8",
    )

    work_dir = tmp_path / "run" / "work"
    input_dir = work_dir / "pool" / "init"
    shutil.copytree(real_init.parents[1], input_dir)
    run_dir = tmp_path / "run"
    result = run_stage1(input_dir, work_dir, run_dir)

    assert result.record_count == 1
    assert result.candidate_count == 1
    assert result.moved_count == 1
    assert source_path.is_file()
    assert not (input_dir / "Genus" / "AA01" / "constant.arc_data").exists()
    assert (work_dir / "delete" / "stage1" / "Genus" / "AA01" / "constant.arc_data").is_file()
    assert (result.stage_dir / "stage1_raw_scores.csv").is_file()
    assert (result.stage_dir / "stage1_candidates.csv").is_file()


def test_plot_commands_generate_explicit_audit_figures(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "alldata"
    init_dir = dataset_dir / "init"
    axis = np.linspace(600.0, 1800.0, 80)
    for folder, offset in (("AA01", 0.0), ("AA02", 0.1)):
        folder_dir = init_dir / "Genus" / folder
        folder_dir.mkdir(parents=True)
        for index in range(3):
            values = np.sin(axis / 100.0) + offset + index * 0.001
            (folder_dir / f"sample_{index}.arc_data").write_text(
                "".join(f"{wavenumber:.3f}\t{value:.6f}\n" for wavenumber, value in zip(axis, values)),
                encoding="utf-8",
            )
    write_delta(dataset_dir / "delta.txt", {("Genus", "AA01"): ("AA", 0.5)})
    monkeypatch.setattr(
        target_plots,
        "resolve_plot_paths",
        lambda dataset_key: PlotPaths(dataset_dir, init_dir, dataset_dir / "fig_init", dataset_dir / "delta.txt"),
    )

    outputs = plot_prefix_dataset(force_enable=True)
    shift_output = plot_shift_folder("alldata", "Genus/AA01")

    assert outputs == [dataset_dir / "fig_init" / "Genus" / "AA.png"]
    assert outputs[0].is_file()
    assert shift_output.is_file()


def test_audit_cli_only_exposes_confirmed_commands():
    parser = build_audit_parser()
    assert parser.parse_args(["clean"]).command == "clean"
    assert parser.parse_args(["plot-prefix"]).command == "plot-prefix"
    assert parser.parse_args(["plot-shift", "--folder", "Genus/AA01"]).command == "plot-shift"
    assert parser.parse_args(["manual-shift", "--folder", "Genus/AA01", "--delta", "0.5"]).command == "manual-shift"


def test_test_pool_copies_only_to_the_audit_run(tmp_path):
    dataset_init = tmp_path / "alldata" / "init" / "Klebsiella" / "KP01"
    dataset_init.mkdir(parents=True)
    (dataset_init / "reference.arc_data").write_text("600\t1\n601\t2\n", encoding="utf-8")
    test_init = tmp_path / "test" / "init"
    mapped = test_init / "CS01KP"
    mapped.mkdir(parents=True)
    (mapped / "sample.arc_data").write_text("600\t3\n601\t4\n", encoding="utf-8")
    unmatched = test_init / "CS02ZZ"
    unmatched.mkdir()
    (unmatched / "skip.arc_data").write_text("600\t5\n601\t6\n", encoding="utf-8")

    run_dir = tmp_path / "alldata" / "audit_runs" / "run"
    run_dir.mkdir(parents=True)
    result = build_work_pool(run_dir, dataset_init.parents[1], test_init)

    target = result.pool_init_dir / "Klebsiella" / "KP__CS01KP__audit" / "sample.arc_data"
    assert result.copied_count == 1
    assert result.skipped_count == 1
    assert target.is_file()
    assert (mapped / "sample.arc_data").is_file()
    assert not (dataset_init.parents[1] / "Klebsiella" / "KP__CS01KP__audit").exists()
    assert (run_dir / "test_pool_manifest.csv").is_file()


def test_test_transfer_builds_grouped_t_folder(tmp_path):
    target_init = tmp_path / "alldata" / "init"
    regular_dir = target_init / "Klebsiella" / "KP01"
    regular_dir.mkdir(parents=True)
    (regular_dir / "regular.arc_data").write_text("600\t1\n", encoding="utf-8")
    prior_dir = target_init / "Klebsiella" / "KP02t"
    prior_dir.mkdir()
    (prior_dir / "prior.arc_data").write_text("600\t9\n", encoding="utf-8")

    source_init = tmp_path / "test" / "init"
    source_dir = source_init / "CS01KP"
    source_dir.mkdir(parents=True)
    for group in ("cell1", "cell2"):
        for repeat in ("000", "001"):
            (source_dir / f"{group}_Area01_{repeat}_shift_cos.arc_data").write_text(
                "600\t1\n", encoding="utf-8"
            )

    manifest = tmp_path / "test" / "test_transfer_manifest.csv"
    folder_map = tmp_path / "test" / "test_transfer_folder_map.csv"
    result = build_test_transfer(
        source_init,
        target_init,
        manifest,
        folder_map,
        TestTransferConfig(train_group_ratio=0.5, random_seed=42),
    )

    target_dir = target_init / "Klebsiella" / "KP02t"
    transferred = sorted(target_dir.glob("*.arc_data"))
    assert result.transferred_count == 2
    assert result.skipped_count == 0
    assert len(transferred) == 2
    assert len({path.name.split("_", 2)[1] for path in transferred}) == 1
    assert manifest.is_file()
    assert folder_map.is_file()
    assert list((tmp_path / "alldata").glob("init_test_transfer_previous_*"))


def test_test_transfer_matches_reference_group_selection(tmp_path):
    source_init = tmp_path / "source" / "init"
    source_dir = source_init / "CS01KP"
    source_dir.mkdir(parents=True)
    for group in ("cell1", "cell2", "cell3"):
        for repeat in ("000", "001"):
            (source_dir / f"{group}_Area01_{repeat}_shift_cos.arc_data").write_text(
                "600\t1\n", encoding="utf-8"
            )
    reference_root = tmp_path / "reference"
    target_root = tmp_path / "target"
    for root in (reference_root, target_root):
        folder = root / "alldata" / "init" / "Klebsiella" / "KP01"
        folder.mkdir(parents=True)
        (folder / "regular.arc_data").write_text("600\t1\n", encoding="utf-8")

    script_path = Path("dataset") / "测试菌" / "transfer_cs_to_init.py"
    module_spec = importlib.util.spec_from_file_location("reference_test_transfer", script_path)
    assert module_spec is not None and module_spec.loader is not None
    reference = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reference)
    reference.SOURCE_ROOT = source_init
    reference.TARGET_INIT = reference_root / "alldata" / "init"
    reference.MANIFEST_PATH = reference_root / "test_transfer_manifest.csv"
    reference.FOLDER_MAP_PATH = reference_root / "test_transfer_folder_map.csv"
    reference.transfer_samples(0.5, 42, False)

    target_manifest = target_root / "test_transfer_manifest.csv"
    target_map = target_root / "test_transfer_folder_map.csv"
    build_test_transfer(
        source_init,
        target_root / "alldata" / "init",
        target_manifest,
        target_map,
        TestTransferConfig(train_group_ratio=0.5, random_seed=42),
    )

    reference_dir = reference_root / "alldata" / "init" / "Klebsiella" / "KP02t"
    target_dir = target_root / "alldata" / "init" / "Klebsiella" / "KP02t"
    assert sorted(path.name for path in target_dir.glob("*.arc_data")) == sorted(
        path.name for path in reference_dir.glob("*.arc_data")
    )
    assert target_manifest.read_bytes() == reference.MANIFEST_PATH.read_bytes()
    assert target_map.read_bytes() == reference.FOLDER_MAP_PATH.read_bytes()

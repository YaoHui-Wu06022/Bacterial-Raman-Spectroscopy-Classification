import json
from pathlib import Path

import numpy as np
import pytest

from ramanv2.audit.heterogeneity import (
    FEATURE_AXIS,
    FolderReport,
    HeterogeneityReport,
    SampleReport,
    VERDICT_MINORITY,
    VERDICT_NO_CLUSTER,
    VERDICT_SPLIT,
    VERDICT_TOO_FEW,
    scan_analysis_folders,
    _classify_folder,
)
from ramanv2.data.catalog import DatasetMapping
from ramanv2.pipeline.anomaly_removal import (
    ANALYSIS_REMOVAL_LOG_NAME,
    ANALYSIS_REMOVAL_REPORT_NAME,
    build_removal_targets,
    check_analysis_gate,
    execute_analysis_removal,
)
from ramanv2.pipeline.lineage import load_folder_lineage
from ramanv2.pipeline.state import COMMIT_LOG_NAME, PipelineState


def write_file(path: Path, text: str = "1 2\n3 4\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def mapping() -> DatasetMapping:
    return DatasetMapping(
        genera={"Escherichia": ("EC",), "Pseudomonas": ("PAE",)},
        datasets={"GN": ("Escherichia", "Pseudomonas")},
    )


def make_sample(
    folder_key: str,
    date_name: str,
    folder_name: str,
    source_kind: str,
    file_name: str,
    *,
    cluster: str = "minority",
    auto: bool = True,
) -> SampleReport:
    return SampleReport(
        sample_id=f"{folder_key}/{file_name}",
        folder_key=folder_key,
        date_name=date_name,
        folder_name=folder_name,
        source_kind=source_kind,
        label="EC",
        genus="Escherichia",
        file_name=file_name,
        order=1,
        features=np.zeros(FEATURE_AXIS.size, dtype=np.float32),
        roughness=0.0,
        cluster=cluster,
        distance_to_centroid=0.1,
        distance_rank=1,
        auto_candidate=auto,
    )


def make_folder(
    folders: list[SampleReport],
    *,
    key: str,
    date_name: str,
    folder_name: str,
    source_kind: str,
    verdict: str = VERDICT_MINORITY,
    separation: float = 0.2,
) -> FolderReport:
    return FolderReport(
        folder_key=key,
        date_name=date_name,
        folder_name=folder_name,
        source_kind=source_kind,
        label="EC",
        genus="Escherichia",
        n_samples=len(folders),
        dispersion=0.1,
        dispersion_ratio=0.6,
        minority_count=sum(1 for sample in folders if sample.cluster == "minority"),
        minority_fraction=0.2,
        cluster_separation=separation,
        verdict=verdict,
        majority_mean=np.zeros(FEATURE_AXIS.size, dtype=np.float32),
        majority_low=np.zeros(FEATURE_AXIS.size, dtype=np.float32),
        majority_high=np.zeros(FEATURE_AXIS.size, dtype=np.float32),
        samples=tuple(folders),
    )


def test_scan_and_classify_folders_ignore_stale_sources(tmp_path: Path) -> None:
    shift = tmp_path / "shift_data"
    supplement = tmp_path / "extra_data"
    write_file(shift / "20240101" / "EC01" / "cell1.arc_data")
    write_file(shift / "20240101" / "EC02" / "cell1.arc_data")
    write_file(shift / "20240101" / "小球" / "bead.arc_data")
    write_file(supplement / "PAE20261104" / "B1.arc_data")

    folders = scan_analysis_folders(
        shift,
        supplement,
        mapping(),
        cosmic_folder_keys={"20240101/EC01"},
    )

    assert [folder.folder_key for folder in folders] == [
        "20240101/EC01",
        "extra/PAE20261104",
    ]
    assert [folder.source_kind for folder in folders] == ["cosmic", "extra"]
    assert folders[1].date_name == "20261104"

    def folder(minority: int, n: int, ratio: float, separation: float) -> FolderReport:
        return FolderReport(
            folder_key="20240101/EC01",
            date_name="20240101",
            folder_name="EC01",
            source_kind="cosmic",
            label="EC",
            genus="Escherichia",
            n_samples=n,
            dispersion=0.1,
            dispersion_ratio=ratio,
            minority_count=minority,
            minority_fraction=minority / n,
            cluster_separation=separation,
        )

    assert _classify_folder(folder(9, 40, 0.665, 0.081)) == VERDICT_MINORITY
    assert _classify_folder(folder(17, 36, 0.923, 0.121)) == VERDICT_SPLIT
    assert _classify_folder(folder(0, 34, 0.746, float("nan"))) == VERDICT_NO_CLUSTER
    assert _classify_folder(folder(0, 4, 1.0, float("nan"))) == VERDICT_TOO_FEW


def test_check_analysis_gate_requires_complete_and_consistent_sources(
    tmp_path: Path,
) -> None:
    cosmic = tmp_path / "cosmic_data"
    shift = tmp_path / "shift_data"
    supplement = tmp_path / "extra_data"
    init = tmp_path / "init"
    write_file(cosmic / "20240101" / "EC01" / "cell1.arc_data")
    write_file(shift / "20240101" / "EC01" / "cell1.arc_data")
    write_file(
        shift / "20240101" / "shift" / "affine.json",
        json.dumps(
            {
                "status": "manual_approved",
                "base_scale": 1.0,
                "base_offset": 0.0,
                "approved_at": "2026-01-01T00:00:00+00:00",
            }
        ),
    )
    init.mkdir()

    gate = check_analysis_gate(
        cosmic,
        shift,
        supplement,
        init,
        mapping(),
        {"20240101": {"EC01"}},
        {"20240101/EC01"},
    )
    assert gate.ready_enable
    assert gate.dates == ("20240101",)
    assert gate.cosmic_folder_keys == ("20240101/EC01",)

    incomplete = check_analysis_gate(
        cosmic,
        shift,
        supplement,
        init,
        mapping(),
        {"20240101": {"EC01"}},
        set(),
    )
    assert not incomplete.ready_enable
    assert any("未完成审核" in issue for issue in incomplete.issues)

    # 小球是校准来源，不参与分析，也不应阻塞清洗完成门禁。
    write_file(cosmic / "20240101" / "小球" / "bead.arc_data")
    bead_pending = check_analysis_gate(
        cosmic,
        shift,
        supplement,
        init,
        mapping(),
        {"20240101": {"EC01", "小球"}},
        {"20240101/EC01"},
    )
    assert bead_pending.ready_enable

    write_file(shift / "20240101" / "EC01" / "stale.arc_data")
    mismatch = check_analysis_gate(
        cosmic,
        shift,
        supplement,
        init,
        mapping(),
        {"20240101": {"EC01"}},
        {"20240101/EC01"},
    )
    assert not mismatch.ready_enable
    assert any("文件不一致" in issue for issue in mismatch.issues)

    extra_cosmic = tmp_path / "extra_cosmic"
    extra_shift = tmp_path / "extra_shift"
    extra_supplement = tmp_path / "extra_supplement"
    extra_init = tmp_path / "extra_init"
    for directory in (extra_cosmic, extra_shift, extra_init):
        directory.mkdir()
    write_file(extra_supplement / "PAE20261104" / "B1.arc_data")

    extra_gate = check_analysis_gate(
        extra_cosmic, extra_shift, extra_supplement, extra_init, mapping(), {}, set()
    )
    assert extra_gate.ready_enable
    assert extra_gate.dates == ()


def test_removal_targets_reject_existing_archive_and_execute_removal(
    tmp_path: Path,
) -> None:
    cosmic = tmp_path / "cosmic_data"
    shift = tmp_path / "shift_data"
    supplement = tmp_path / "extra_data"
    init = tmp_path / "init"
    write_file(cosmic / "20240101" / "EC01" / "cell1.arc_data")
    write_file(shift / "20240101" / "EC01" / "cell1.arc_data")
    write_file(supplement / "PAE20261104" / "B1.arc_data")
    write_file(init / "Escherichia" / "EC01" / "cell1.arc_data")

    cosmic_sample = make_sample(
        "20240101/EC01", "20240101", "EC01", "cosmic", "cell1.arc_data"
    )
    extra_sample = make_sample(
        "extra/PAE20261104", "20261104", "PAE20261104", "extra", "B1.arc_data"
    )
    cosmic_folder = make_folder(
        [cosmic_sample],
        key="20240101/EC01",
        date_name="20240101",
        folder_name="EC01",
        source_kind="cosmic",
    )
    extra_folder = make_folder(
        [extra_sample],
        key="extra/PAE20261104",
        date_name="20261104",
        folder_name="PAE20261104",
        source_kind="extra",
    )
    report = HeterogeneityReport(
        folders=(cosmic_folder, extra_folder),
        drift_threshold=0.5,
        generated_at="test",
    )

    targets = build_removal_targets(
        cosmic,
        shift,
        supplement,
        report,
        [cosmic_sample.sample_id],
    )
    assert targets[0].archive_path == (
        tmp_path / "analysis_delete" / "20240101" / "EC01" / "cell1.arc_data"
    )

    write_file(targets[0].archive_path)
    with pytest.raises(FileExistsError):
        build_removal_targets(cosmic, shift, supplement, report, [cosmic_sample.sample_id])
    targets[0].archive_path.unlink()

    state = PipelineState(
        init_generated_dates={"20240101"},
        dataset_builds={
            "full_init": {"status": "ready", "pending_dates": []},
            "latest": {"status": "ready"},
        },
    )
    state_path = tmp_path / ".raman_pipeline" / "state.json"

    result = execute_analysis_removal(
        cosmic,
        shift,
        supplement,
        state,
        state_path,
        report,
        [cosmic_sample.sample_id, extra_sample.sample_id],
    )

    assert result["sample_count"] == 2
    assert result["cosmic_count"] == 1
    assert result["extra_count"] == 1
    archive = tmp_path / "analysis_delete"
    assert (archive / "20240101" / "EC01" / "cell1.arc_data").is_file()
    assert (archive / "20240101" / "EC01" / "cell1.png").is_file()
    assert (archive / "20261104" / "PAE20261104" / "B1.arc_data").is_file()
    assert (archive / "20261104" / "PAE20261104" / "B1.png").is_file()
    assert not (cosmic / "20240101" / "EC01" / "cell1.arc_data").exists()
    assert not (shift / "20240101" / "EC01" / "cell1.arc_data").exists()
    assert not (supplement / "PAE20261104" / "B1.arc_data").exists()
    assert (init / "Escherichia" / "EC01" / "cell1.arc_data").is_file()

    lineage = load_folder_lineage(cosmic)
    entry = lineage["folders"]["20240101/EC01"]["files"]["cell1.arc_data"]
    assert entry["status"] == "analysis_deleted"
    assert entry["analysis_deleted_path"] == (
        "analysis_delete/20240101/EC01/cell1.arc_data"
    )
    assert state.init_generated_dates == set()
    assert state.dataset_builds["full_init"]["status"] == "pending"
    assert "latest" not in state.dataset_builds

    commit_log = state_path.parent / COMMIT_LOG_NAME
    removal_log = state_path.parent / ANALYSIS_REMOVAL_LOG_NAME
    assert json.loads(commit_log.read_text(encoding="utf-8").splitlines()[-1])[
        "operation"
    ] == "analysis_anomaly_removal"
    assert json.loads(removal_log.read_text(encoding="utf-8").splitlines()[-1])[
        "sample_count"
    ] == 2

    report_path = archive / ANALYSIS_REMOVAL_REPORT_NAME
    assert result["report_path"] == f"analysis_delete/{ANALYSIS_REMOVAL_REPORT_NAME}"
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "## 当前归档" in report_text
    assert "## 最近一次剔除" in report_text
    assert "| 批次时间 | 合计 | cosmic | extra | 涉及文件夹 |" in report_text
    assert "shift 删除" not in report_text.split("## 历史剔除批次")[1]
    assert "光谱：2 条" in report_text
    assert "合计：2 条（cosmic 1 / extra 1）" in report_text
    assert "| 20240101/EC01 | cosmic | 1 |" in report_text
    assert "| extra/PAE20261104 | extra | 1 |" in report_text

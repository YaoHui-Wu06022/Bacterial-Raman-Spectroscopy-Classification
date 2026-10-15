"""页面与模块职责的轻量结构约束。

只保留「改错就会静默破坏数据流」的少量断言，不做逐标签的源码字符串核对。
"""

from pathlib import Path


def test_ui_pages_are_independent_importable_modules() -> None:
    from ramanv2.ui.app import PAGE_NAMES, main
    from ramanv2.ui.pages.affine_calibration import render_affine_calibration_page
    from ramanv2.ui.pages.cleaning import render_cleaning_page
    from ramanv2.ui.pages.dataset_acquisition import render_dataset_acquisition_page
    from ramanv2.ui.pages.hierarchical_training import render_hierarchical_training_page
    from ramanv2.ui.pages.result_analysis import render_result_analysis_page
    from ramanv2.ui.pages.sample_inference import render_sample_inference_page

    assert PAGE_NAMES == ("数据清洗", "仿射校准", "数据集获取", "层级训练", "结果分析", "样本推理")
    assert all(
        callable(page)
        for page in (
            main,
            render_cleaning_page,
            render_affine_calibration_page,
            render_dataset_acquisition_page,
            render_hierarchical_training_page,
            render_result_analysis_page,
            render_sample_inference_page,
        )
    )


def test_pipeline_modules_keep_data_flow_separation() -> None:
    """恢复只改 cosmic、审核只做一次、shift 只由整夹重建生成。"""
    cleaning = Path("ramanv2/pipeline/cleaning.py").read_text(encoding="utf-8")
    restoration = Path("ramanv2/pipeline/restoration.py").read_text(encoding="utf-8")
    affine_page = Path("ramanv2/ui/pages/affine_calibration.py").read_text(encoding="utf-8")

    # 清洗不再拒绝已校准日期的源谱改动（否则整夹恢复后无法重新清洗）。
    assert "清洗页不能继续修改源谱" not in cleaning
    # 恢复不写 shift/init、不作废审核。
    assert "write_arc_data" not in restoration
    assert "invalidate_audit" not in restoration
    assert "run_folder_audit" not in restoration
    # 审核直接对当前 shift 正本评分，不归位；已完成日期跳过。
    # shift 重建入口在仿射校准页，且是整夹重建。
    assert "rebuild_shift_data" in affine_page
    assert "scan_shift_pending_folders" not in affine_page
    # 重建不再是独立入口：已并入「一键分析并校准」，页面上只有一个按钮。
    assert "_render_shift_rebuild_section" not in affine_page
    assert "整夹重建 shift 数据" not in affine_page


def test_one_click_calibrates_missing_and_reuses_existing(tmp_path, monkeypatch) -> None:
    """一键流程：缺参数的日期走分析，已有参数的日期复用参数重建 shift。"""
    from types import SimpleNamespace

    from ramanv2.ui.pages import affine_calibration as page

    calls: dict[str, list[str]] = {}

    def fake_calibration(source_root, date_names, output_root):
        calls["calibration"] = list(date_names)
        return [{"date": name, "status": "manual_approved"} for name in date_names]

    def fake_rebuild(data_root, shift_root, date_names):
        calls["rebuild"] = list(date_names)
        return [{"date": name, "status": "rebuilt", "spectrum_count": 3} for name in date_names]

    monkeypatch.setattr(page, "run_affine_calibration", fake_calibration)
    monkeypatch.setattr(page, "rebuild_shift_data", fake_rebuild)
    context = SimpleNamespace(data_root=tmp_path / "cosmic_data", shift_root=tmp_path / "shift_data")

    reports = page._run_analyze_and_rebuild(context, ["20240102"], ["20240101"])

    assert calls == {"calibration": ["20240102"], "rebuild": ["20240101"]}
    assert reports["calibration"][0]["status"] == "manual_approved"
    assert reports["rebuild"][0]["spectrum_count"] == 3


def test_one_click_skips_empty_groups(tmp_path, monkeypatch) -> None:
    """没有待校准日期时不重复分析小球；没有已校准日期时不重建。"""
    from types import SimpleNamespace

    from ramanv2.ui.pages import affine_calibration as page

    calls: list[str] = []

    def fake_calibration(source_root, date_names, output_root):
        calls.append("calibration")
        return []

    def fake_rebuild(data_root, shift_root, date_names):
        calls.append("rebuild")
        return []

    monkeypatch.setattr(page, "run_affine_calibration", fake_calibration)
    monkeypatch.setattr(page, "rebuild_shift_data", fake_rebuild)
    context = SimpleNamespace(data_root=tmp_path / "cosmic_data", shift_root=tmp_path / "shift_data")

    assert page._run_analyze_and_rebuild(context, [], ["20240101"]) == {
        "calibration": [],
        "rebuild": [],
    }
    assert calls == ["rebuild"]

    calls.clear()
    page._run_analyze_and_rebuild(context, ["20240102"], [])
    assert calls == ["calibration"]


def test_dataset_page_only_defaults_unmapped_cs_folders() -> None:
    from ramanv2.data.catalog import DatasetMapping
    from ramanv2.ui.pages.dataset_acquisition import _default_unmapped_cs_sources

    mapping = DatasetMapping.from_payload(
        {"genera": {"Klebsiella": ["KP"]}, "datasets": {"GN": ["Klebsiella"]}}
    )
    assert _default_unmapped_cs_sources(
        ["20240101/CS01KP", "20240101/CS02AA", "20240101/CS03"],
        mapping,
    ) == ["20240101/CS02AA", "20240101/CS03"]


def test_training_page_builds_existing_train_request() -> None:
    from ramanv2.ui.pages.hierarchical_training import _build_train_request

    request = _build_train_request(
        {"profile_id": "GN", "epochs": 2, "batch_size": 4},
        {
            "level_name": "level_1",
            "train_per_parent_enable": False,
            "only_parent_name": None,
        },
        {"experiment_dir": None, "run_name": "smoke", "resume_run_dir": None},
    )

    assert request.config.dataset.profile_id == "GN"
    assert request.config.training.epochs == 2
    assert request.level_name == "level_1"
    assert request.train_per_parent_enable is False


def test_one_click_call_site_matches_signature() -> None:
    """按钮回调必须按 `_run_analyze_and_rebuild` 的形参个数传参。

    直接调用函数的用例无法发现调用点多传/少传参数（点击才炸），这里用 AST 静态核对。
    """
    import ast
    import inspect

    from ramanv2.ui.pages import affine_calibration as page

    target = page._run_analyze_and_rebuild
    expected = len(inspect.signature(target).parameters)
    tree = ast.parse(Path("ramanv2/ui/pages/affine_calibration.py").read_text(encoding="utf-8"))
    call_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_analyze_and_rebuild"
    ]
    assert call_sites, "仿射页应至少有一处调用 _run_analyze_and_rebuild"
    for node in call_sites:
        assert not node.keywords, "应使用位置参数调用"
        assert len(node.args) == expected, (
            f"第 {node.lineno} 行传了 {len(node.args)} 个参数，签名要求 {expected} 个"
        )


def test_affine_page_only_returns_incremental_dates() -> None:
    from ramanv2.ui.pages.affine_calibration import _pending_date_names

    assert _pending_date_names(
        ["20240101", "20240102", "20240103"],
        {"20240101", "20240102"},
    ) == ["20240103"]


def test_affine_gate_requires_active_source_folder(tmp_path) -> None:
    from ramanv2.pipeline.lineage import has_active_source_folders

    date_dir = tmp_path / "20240101"
    (date_dir / "小球").mkdir(parents=True)
    (date_dir / "EC01" / "cell.arc_data").parent.mkdir(parents=True)
    (date_dir / "EC01" / "cell.arc_data").write_text("1 2\n", encoding="utf-8")
    assert has_active_source_folders(tmp_path, "20240101")

    archive = tmp_path / "delete" / "folder_delete" / "20240101"
    archive.mkdir(parents=True)
    (date_dir / "EC01").rename(archive / "EC01")
    assert not has_active_source_folders(tmp_path, "20240101")

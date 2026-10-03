def test_ui_pages_are_independent_importable_modules() -> None:
    from ramanv2.ui.app import PAGE_NAMES, main
    from ramanv2.ui.pages.affine_calibration import render_affine_calibration_page
    from ramanv2.ui.pages.cleaning import render_cleaning_page
    from ramanv2.ui.pages.dataset_acquisition import render_dataset_acquisition_page
    from ramanv2.ui.pages.hierarchical_training import render_hierarchical_training_page
    from ramanv2.ui.pages.result_analysis import render_result_analysis_page
    from ramanv2.ui.pages.sample_inference import render_sample_inference_page

    assert PAGE_NAMES == ("数据清洗", "仿射校准", "数据集获取", "层级训练", "结果分析", "样本推理")
    assert "流程分页" not in PAGE_NAMES
    assert all(callable(page) for page in (
        main,
        render_cleaning_page,
        render_affine_calibration_page,
        render_dataset_acquisition_page,
        render_hierarchical_training_page,
        render_result_analysis_page,
        render_sample_inference_page,
    ))


def test_cleaning_toolbar_is_incremental() -> None:
    from pathlib import Path

    source = Path("ramanv2/ui/pages/cleaning.py").read_text(encoding="utf-8")
    for label in ("上一页", "下一页", "文件夹移除", "日期移除"):
        assert label in source
    for label in ("开始下一批", "标记已选移除", "清除已选", "标记文件夹已审核", "撤销标记"):
        assert label not in source


def test_dataset_page_has_separate_full_init_and_profile_actions() -> None:
    from pathlib import Path

    source = Path("ramanv2/ui/pages/dataset_acquisition.py").read_text(encoding="utf-8")
    assert "从 shift_data 重建全量 init" in source
    assert "从 init 生成数据集" in source
    assert "生成 init 和数据集子集" not in source


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


def test_date_removal_rescans_disk_before_moving_files() -> None:
    from pathlib import Path

    source = Path("ramanv2/ui/pages/cleaning.py").read_text(encoding="utf-8")
    assert "force_enable=True" in source
    assert "date_catalog.values()" in source


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


def test_training_page_uses_click_only_chinese_controls() -> None:
    from pathlib import Path

    source = Path("ramanv2/ui/pages/hierarchical_training.py").read_text(encoding="utf-8")
    assert "text_input" not in source
    for label in ("开始本地训练", "生成 data.zip 和 ramanv2.zip"):
        assert label in source
    for label in ("训练方案", "标准训练", "快速试跑", "低显存训练"):
        assert label not in source


def test_affine_page_only_returns_incremental_dates() -> None:
    from ramanv2.ui.pages.affine_calibration import _pending_date_names

    assert _pending_date_names(
        ["20240101", "20240102", "20240103"],
        {"20240101", "20240102"},
        {"20240101"},
    ) == ["20240102", "20240103"]

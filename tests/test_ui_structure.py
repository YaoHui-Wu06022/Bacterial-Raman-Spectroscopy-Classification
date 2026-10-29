"""页面与模块职责的轻量结构约束。

只保留「改错就会静默破坏数据流」的少量断言，不做逐标签的源码字符串核对。
"""



def test_ui_pages_are_independent_importable_modules() -> None:
    from ramanv2.ui.app import PAGE_NAMES, main
    from ramanv2.ui.page_views.affine_calibration import render_affine_calibration_page
    from ramanv2.ui.page_views.cleaning import render_cleaning_page
    from ramanv2.ui.page_views.dataset_acquisition import render_dataset_acquisition_page
    from ramanv2.ui.page_views.hierarchical_training import render_hierarchical_training_page
    from ramanv2.ui.page_views.result_analysis import render_result_analysis_page
    from ramanv2.ui.page_views.sample_inference import render_sample_inference_page

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



















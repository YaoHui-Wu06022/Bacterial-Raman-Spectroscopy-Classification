"""训练结果分析入口"""

from raman.analysis import (
    HeatmapConfig,
    run_analysis_cascade,
    run_analysis_level_only,
    run_analysis_single_model,
)


# 可选模式：single_model / level_only / cascade
ANALYSIS_MODE = "single_model"

# single_model 模式填具体 run_* 或 best/run_* 目录
RUN_DIR = ""

# level_only / cascade 模式填实验根目录
EXP_DIR = ""

TARGET_LEVEL = "level_1"
PARENT_IDX = None
INHERIT_MISSING_LEVELS = False

HEATMAP_NUM_BATCHES = 10
HEATMAP_STEPS = 32
HEATMAP_MAX_PER_CLASS = 50
HEATMAP_ROW_NORM = "max"
HEATMAP_USE_TRAIN_LOADER = True
HEATMAP_TOPK_PER_CLASS = 5


def _heatmap_config():
    return HeatmapConfig(
        num_batches=HEATMAP_NUM_BATCHES,
        steps=HEATMAP_STEPS,
        max_per_class=HEATMAP_MAX_PER_CLASS,
        row_norm=HEATMAP_ROW_NORM,
        use_train_loader=HEATMAP_USE_TRAIN_LOADER,
        topk_per_class=HEATMAP_TOPK_PER_CLASS,
    )


def main():
    heatmap_cfg = _heatmap_config()
    if ANALYSIS_MODE == "single_model":
        if not RUN_DIR:
            raise ValueError("single_model 模式请先填写 RUN_DIR")
        result_dir = run_analysis_single_model(
            RUN_DIR,
            level=TARGET_LEVEL,
            parent_idx=PARENT_IDX,
            inherit_missing_levels=INHERIT_MISSING_LEVELS,
            heatmap_cfg=heatmap_cfg,
        )
    elif ANALYSIS_MODE == "level_only":
        if not EXP_DIR:
            raise ValueError("level_only 模式请先填写 EXP_DIR")
        result_dir = run_analysis_level_only(
            EXP_DIR,
            TARGET_LEVEL,
            parent_idx=PARENT_IDX if PARENT_IDX is not None else "all",
            inherit_missing_levels=INHERIT_MISSING_LEVELS,
            heatmap_cfg=heatmap_cfg,
        )
    elif ANALYSIS_MODE == "cascade":
        if not EXP_DIR:
            raise ValueError("cascade 模式请先填写 EXP_DIR")
        result_dir = run_analysis_cascade(
            EXP_DIR,
            TARGET_LEVEL,
            parent_idx=PARENT_IDX if PARENT_IDX is not None else "all",
            inherit_missing_levels=INHERIT_MISSING_LEVELS,
            heatmap_cfg=heatmap_cfg,
        )
    else:
        raise ValueError("ANALYSIS_MODE 只能是 single_model / level_only / cascade")

    print("analysis_result_dir =", result_dir)


if __name__ == "__main__":
    main()

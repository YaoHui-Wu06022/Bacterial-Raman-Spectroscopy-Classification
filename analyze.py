from raman.analysis import AnalysisOverrides, HeatmapConfig, run_analysis_pipeline


# 手动配置
EXP_DIR = ""  # 训练输出目录（含模型与层级元数据）
ANALYSIS_MODE = "single"  # single / aggregate
ANALYSIS_LEVEL = "level_1"  # 分析层级（None 时用当前训练层级）
PARENT_IDX = None  # single: None / 指定 id / "all"；aggregate: 通常用 "all"
USE_TRAIN_AUG = False  # 是否使用训练增强来构建分析数据
FALLBACK_TO_SINGLE = True  # 聚合模式下没有 parent 子模型时，是否退化为单模型分析
INHERIT_MISSING_LEVELS = True  # 缺失层级时向最低级继承（便于展示）

HEATMAP_NUM_BATCHES = 10  # 采样多少个 batch 计算热图
HEATMAP_STEPS = 32  # IG 积分步数（越大越稳但更慢）
HEATMAP_MAX_PER_CLASS = 50  # 每类最多使用的样本数上限
HEATMAP_TARGET_MODE = "true"  # true=真实标签，pred=预测标签
HEATMAP_ROW_NORM = "max"  # 行归一化方式：max / sum / none
HEATMAP_USE_TRAIN_LOADER = True  # True 用训练集，False 用测试集
HEATMAP_TOPK_PER_CLASS = 5  # 每类导出 top-k 波段


def main():
    heatmap_cfg = HeatmapConfig(
        num_batches=HEATMAP_NUM_BATCHES,
        steps=HEATMAP_STEPS,
        max_per_class=HEATMAP_MAX_PER_CLASS,
        target_mode=HEATMAP_TARGET_MODE,
        row_norm=HEATMAP_ROW_NORM,
        use_train_loader=HEATMAP_USE_TRAIN_LOADER,
        topk_per_class=HEATMAP_TOPK_PER_CLASS,
    )
    run_analysis_pipeline(
        overrides=AnalysisOverrides(
            exp_dir=EXP_DIR,
            mode=ANALYSIS_MODE,
            analysis_level=ANALYSIS_LEVEL,
            parent_idx=PARENT_IDX,
            use_train_aug=USE_TRAIN_AUG,
            inherit_missing_levels=INHERIT_MISSING_LEVELS,
            fallback_to_single=FALLBACK_TO_SINGLE,
        ),
        heatmap_cfg=heatmap_cfg,
    )


if __name__ == "__main__":
    main()

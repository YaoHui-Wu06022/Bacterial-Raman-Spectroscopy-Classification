try:
    from . import analysis_core as ac
except ImportError:
    import analysis_core as ac


# ---------------- 手动配置 ----------------
EXP_DIR = ""  # 训练输出目录（含模型与层级元数据）
ANALYSIS_LEVEL = "level_2"              # 分析层级（聚合通常是 level_2）
PARENT_IDX = "all"                      # 参与聚合的 parent：all / 指定 id / None
USE_TRAIN_AUG = False                   # 是否使用训练增强来构建分析数据

HEATMAP_NUM_BATCHES = 10                # 采样多少个 batch 计算热图
HEATMAP_STEPS = 32                      # IG 积分步数（越大越稳但更慢）
HEATMAP_MAX_PER_CLASS = 50              # 每类最多使用的样本数上限
HEATMAP_TARGET_MODE = "true"            # true=真实标签，pred=预测标签
HEATMAP_ROW_NORM = "max"                # 行归一化方式：max / sum / none
HEATMAP_USE_TRAIN_LOADER = True         # True 用训练集，False 用测试集
HEATMAP_TOPK_PER_CLASS = 5              # 每类导出 top-k 波段
INHERIT_MISSING_LEVELS = True           # 缺失层级时向最低级继承（便于展示）


def main():
    heatmap_cfg = ac.HeatmapConfig(
        num_batches=HEATMAP_NUM_BATCHES,
        steps=HEATMAP_STEPS,
        max_per_class=HEATMAP_MAX_PER_CLASS,
        target_mode=HEATMAP_TARGET_MODE,
        row_norm=HEATMAP_ROW_NORM,
        use_train_loader=HEATMAP_USE_TRAIN_LOADER,
        topk_per_class=HEATMAP_TOPK_PER_CLASS,
    )
    ac.run_aggregate_pipeline(
        EXP_DIR,
        ANALYSIS_LEVEL,
        PARENT_IDX,
        use_train_aug=USE_TRAIN_AUG,
        heatmap_cfg=heatmap_cfg,
        inherit_missing_levels=INHERIT_MISSING_LEVELS,
        fallback_to_single=True,
    )


if __name__ == "__main__":
    main()

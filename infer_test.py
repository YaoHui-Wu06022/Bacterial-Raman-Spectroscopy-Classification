"""独立测试集推理入口"""

from dataclasses import replace

import torch

from raman.data.build import build_test
from raman.data.profiles import get_dataset_dir, get_profile
from raman.infer.test import run_independent_test


# 手动配置
EXP_DIR = "output/GN/20260722_051934_div4_89%"
LEVEL = "level_2"

# 可选配置
BUILD_TEST_FROM_INIT = True  # True 时先由 测试菌/init 重建 测试菌/test
TEST_ROOT = "dataset/测试菌/test"  # 关闭上方开关时使用的测试数据目录
FOLDER = None  # 例如 "CS01KP"，None 表示运行全部测试文件夹
TOP_K = 3  # 每条光谱输出置信度最高的前 k 个预测类别
USE_CPU = False  # False 表示优先使用 CUDA，不可用时自动回退到 CPU
EVALUATE_EXPECTED_LABEL = True  # 根据测试文件夹前缀计算准确率和混淆矩阵
PLOT_TRAIN_MEAN = False  # 是否读取 train 并在谱图中额外绘制训练集均值作对照
SKIP_TRANSFERRED_TEST_SAMPLES = True  # 跳过已插入训练集的测试谱，避免数据泄漏
TRANSFER_MANIFEST = "dataset/测试菌/test_transfer_manifest.csv"  # 已迁移测试谱清单


def main():
    """运行独立测试集推理"""
    if not EXP_DIR:
        raise ValueError("请先在 infer_test.py 里填写 EXP_DIR")

    if BUILD_TEST_FROM_INIT:
        profile = replace(get_profile("test"), root_init_test="init")
        dataset_dir = get_dataset_dir(profile)
        build_test(profile, dataset_dir)

    device = torch.device("cpu") if USE_CPU else None
    run_independent_test(
        EXP_DIR,
        LEVEL,
        test_root=TEST_ROOT,
        folder=FOLDER,
        top_k=TOP_K,
        device=device,
        evaluate=EVALUATE_EXPECTED_LABEL,
        plot_train_mean=PLOT_TRAIN_MEAN,
        skip_transferred=SKIP_TRANSFERRED_TEST_SAMPLES,
        transfer_manifest=TRANSFER_MANIFEST,
    )


if __name__ == "__main__":
    main()

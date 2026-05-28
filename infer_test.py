"""独立测试集推理入口"""

import torch

from raman.infer.test import run_independent_test


# 手动配置
EXP_DIR = "output/GN/20260528_120101"
LEVEL = "level_1"

# 可选配置
TEST_ROOT = None  # None 表示使用模型配置对应数据集的 test
FOLDER = None  # 例如 "CS01KP"，None 表示运行全部测试文件夹
TOP_K = 3
USE_CPU = False
SKIP_TRANSFERRED_TEST_SAMPLES = True
TRANSFER_MANIFEST = "dataset/测试菌/test_transfer_manifest.csv"


def main():
    """运行独立测试集推理"""
    if not EXP_DIR:
        raise ValueError("请先在 infer_test.py 里填写 EXP_DIR")

    device = torch.device("cpu") if USE_CPU else None
    run_independent_test(
        EXP_DIR,
        LEVEL,
        test_root=TEST_ROOT,
        folder=FOLDER,
        top_k=TOP_K,
        device=device,
        skip_transferred=SKIP_TRANSFERRED_TEST_SAMPLES,
        transfer_manifest=TRANSFER_MANIFEST,
    )


if __name__ == "__main__":
    main()

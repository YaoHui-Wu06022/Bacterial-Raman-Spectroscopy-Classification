"""独立测试集推理入口"""

from dataclasses import fields, replace

import torch

from raman.config_io import load_experiment
from raman.data.build import DEFAULT_PIPELINE_CONFIG, build_test
from raman.data.profiles import get_dataset_dir, get_profile
from raman.infer.test import run_independent_test


# 手动配置
EXP_DIR = "output/肠杆菌/八分类去K/20260523_124451"
LEVEL = "level_1"

# 可选配置
BUILD_TEST_FIRST = False  # True 时先按模型配置从 init_test 生成 test
TEST_ROOT = None  # None 表示使用模型配置对应数据集的 test
FOLDER = None  # 例如 "CS01KP"，None 表示运行全部测试文件夹
TOP_K = 3
USE_CPU = False
SKIP_TRANSFERRED_TEST_SAMPLES = True
TRANSFER_MANIFEST = "dataset/细菌/test_transfer_manifest.csv"


def _pipeline_config_from_model(config):
    """用模型 config.yaml 里保存的预处理参数覆盖当前默认值"""
    updates = {}
    for item in fields(DEFAULT_PIPELINE_CONFIG):
        if not hasattr(config, item.name):
            continue
        value = getattr(config, item.name)
        if item.name == "bad_bands":
            value = tuple(tuple(float(x) for x in band) for band in (value or ()))
        updates[item.name] = value
    return replace(DEFAULT_PIPELINE_CONFIG, **updates)


def main():
    """运行独立测试集推理"""
    if not EXP_DIR:
        raise ValueError("请先在 infer_test.py 里填写 EXP_DIR")

    config = load_experiment(EXP_DIR)
    profile = get_profile(config.dataset_name)
    dataset_dir = get_dataset_dir(profile)
    if BUILD_TEST_FIRST:
        build_test(
            profile,
            dataset_dir,
            pipeline_config=_pipeline_config_from_model(config),
        )

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

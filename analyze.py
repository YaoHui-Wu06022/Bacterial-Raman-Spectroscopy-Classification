"""在 PyCharm 或 Colab 中直接运行的模型解释入口。"""

from ramanv2.analysis.runner import run_interpret_parent_routed, run_interpret_run


# MODE 取 run 或 parent_routed；分析默认参数在 ramanv2/core/config.py 中维护。
MODE = "run"
SOURCE_DIR = ""
LEVEL_NAME = "level_1"
PARENT = None
CPU_ENABLE = False


def main():
    """按单 run 或真实父类路由方式执行解释性分析。"""
    if not SOURCE_DIR:
        raise ValueError("请先在 analyze.py 里填写 SOURCE_DIR")

    device = "cpu" if CPU_ENABLE else None
    if MODE == "run":
        result_dir = run_interpret_run(SOURCE_DIR, LEVEL_NAME, device)
    elif MODE == "parent_routed":
        result_dir = run_interpret_parent_routed(
            SOURCE_DIR,
            LEVEL_NAME,
            PARENT,
            device,
        )
    else:
        raise ValueError("MODE 只能是 run / parent_routed")

    print("analysis_result_dir =", result_dir)


if __name__ == "__main__":
    main()

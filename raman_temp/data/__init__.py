"""常规数据集的 profile、I/O、运行时视图和离线构建流程。

为使根级 ``--help`` 不加载 PyTorch 或数值处理依赖，调用方应从具体子模块
导入所需对象，例如 ``raman_temp.data.dataset.RamanDataset``。
"""

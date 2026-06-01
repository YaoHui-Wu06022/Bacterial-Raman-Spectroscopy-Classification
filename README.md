# 拉曼光谱层级分类项目

## 1. 项目目标

本项目面向微生物拉曼光谱识别任务，构建一套完整的层级分类实验系统

当前代码已经覆盖：

- 原始 `.arc_data` 的离线清洗与目录重组
- 面向 1D 光谱的在线预处理、多通道输入构建与数据增强
- 逐层级、按父类拆分的训练流程
- 独立测试集对照分析、传统机器学习基线对照
- 训练后可解释性分析与测试集 embedding 诊断

数据集目录涵盖：

- 属级标签，例如 `Escherichia / Klebsiella / Proteus`
- 更细一级的叶级标签，例如 `EC / KP / PMI`

训练时，系统会先从目录结构自动构建层级标签树，再按当前训练层级决定训练目标：

- 顶层全局模型
- 或某一层下按父类拆开的子模型

项目当前的技术主线如下：

1. 离线预处理阶段：做宇宙射线去除、基线校正、波段裁剪、坏段剔除、统一波数轴插值、训练集 PCA 异常值过滤
2. 在线输入阶段：从清洗后的单通道光谱构造模型输入
3. 模型阶段：使用多尺度 1D CNN 主干提取局部峰形，再接序列编码器和分类头
4. 训练阶段：围绕层级标签、类别不均衡和细粒度难样本设计多种损失与重加权策略
5. 评估与分析阶段：通过混淆矩阵、embedding 近邻诊断、IG、Layer Grad-CAM 等方式分析错误来源

## 2. 仓库结构与模块职责

### 2.1 根目录入口

```text
拉曼光谱分类/
├─ train.py                          # 训练入口，只负责手动覆盖项与 run_training 调用
├─ evaluate.py                       # 独立测试集评估入口
├─ infer_test.py                     # 独立测试集推理入口，手动改顶部配置后运行
├─ pca_svm_baseline.py               # PCA + SVM 基线评估入口
├─ analyze.py                        # 分析入口，支持 single_model / level_only / cascade 三种模式
├─ AGENTS.md                         # 本仓库给 coding agent 的协作约束
├─ LICENSE                           # 项目许可证
└─ README.md                         # 项目流程、参数和方法说明
```

### 2.2 核心模型与配置

```text
raman/
├─ config.py                         # 训练配置定义：输入、模型、损失、增强概率、优化器参数
├─ config_io.py                      # config.yaml 读写与实验配置回载
├─ model.py                          # 主模型实现：多尺度 stem + 1D CNN + encoder + pooling + head
├─ trainer.py                        # 训练总调度：数据集、层级任务、模型循环与 hierarchy meta
└─ __init__.py                       # raman 包标记
```

### 2.3 数据层

```text
raman/data/
├─ loader.py                         # RamanDataset：目录扫描、层级标签、样本索引与读取接口
├─ input.py                          # 在线模型输入：标准化、SG、d1 通道与训练增强
├─ io.py                             # .arc_data 读写、init.npz 打包/解包与 init 输入解析
├─ preprocess.py                     # 离线基线校正、宇宙射线去除、单谱清洗与均值谱绘图
├─ build.py                          # 离线构建主流程：build_train、build_test
├─ plot.py                           # 从已有 train 独立生成 fig_train 审图
├─ count.py                          # 数据集文件数量统计与树形输出
├─ profiles.py                       # 各数据集的目录布局、数据集名称与别名
├─ cli.py                            # 离线数据处理 CLI：pack/unpack/train/test/plot/count
└─ __main__.py                       # 支持 python -m raman.data

raman/shift/
├─ cli.py                            # 原始谱 preview、单文件夹平移与平移前后对照 CLI
├─ core.py                           # 同属同前缀审图、delta 记录、迁移目录快照与波数列平移
└─ __main__.py                       # 支持 python -m raman.shift
```

### 2.4 通用工具层

```text
raman/tool/
├─ array.py                          # 一维窗口、连续区间、移动平均与鲁棒尺度工具
├─ dataset.py                        # 数据集 profile 解析、阶段目录解析与 .arc_data 叶目录遍历
├─ hierarchy.py                      # 层级 key、level 名称、hierarchy_meta 读写与规整
├─ model.py                          # 模型输出 logits 选择与 RNN 反传保护判断
├─ naming.py                         # 类别前缀、测试文件夹前缀与文件名前缀工具
├─ path.py                           # 项目根目录、相对路径规范化和安全路径解析
├─ plotting.py                       # 坏段断线绘图、灰色坏段标注与混淆矩阵尺寸辅助
├─ spectrum.py                       # 波数轴、坏段 mask、坏段规范化与输出波数轴工具
└─ __init__.py                       # raman.tool 包标记
```

`raman/tool` 只放跨数据、审核、训练、评估、推理和分析复用的轻量工具；具体业务流程仍保留在各自子包中。

### 2.5 审核层

```text
raman/audit/
├─ cli.py                            # 审核 CLI：full / bad-band / move
├─ full_scan.py                      # 分阶段全库审核入口与报告、复核图输出
├─ config.py                         # AuditConfig：各阶段阈值、删除分类与原因标签
├─ scoring.py                        # 审核公共评分、SpectrumRecord 字段整理与阶段调度
├─ common.py                         # 审核载荷预处理、记录读取与 CSV 输出
├─ stage.py                          # invalid / class-similarity 两阶段判定规则
├─ bad_band.py                       # 系统性下凹坏段只读扫描
├─ move.py                           # 按路径或候选清单移动到 delete
└─ __main__.py                       # 支持 python -m raman.audit
```

### 2.6 训练层

```text
raman/training/
├─ split.py                          # 训练/验证切分、训练范围解析、父类过滤
├─ losses.py                         # Focal / SupCon / AlignLoss / 类别权重
├─ model_loop.py                     # 单个模型的训练循环、EMA 权重、最佳模型保存
├─ validation.py                     # 训练期验证循环、父类 mask、局部标签映射与指标计算
├─ checkpoint.py                     # 续训 checkpoint 保存、恢复与清理
├─ se_stats.py                       # 训练期累计 SEBlock 缩放统计
└─ session.py                        # 输出目录、日志、随机种子与配置快照
```

### 2.7 评估与推理层

```text
raman/eval/
├─ experiment.py                     # 实验目录、配置、hierarchy meta 与模型路径解析
├─ runtime.py                        # 实验运行时：模型懒加载、缓存与 SE sidecar 读取
├─ common.py                         # 共享推理辅助：logits 选择、层级掩码、级联推理、指标计算
├─ evaluator.py                      # 测试集评估主流程与结果落盘
├─ baseline.py                       # PCA + SVM 基线实现
└─ report.py                         # classification report、混淆矩阵、文本结果输出

raman/infer/
├─ core.py                           # 层级级联推理核心
├─ folder.py                         # 文件夹逐谱预测、top-k 文本报告与文件枚举
├─ labels.py                         # 测试文件夹标签映射、文件夹多数投票和 summary 输出
├─ spectra.py                        # 测试谱输入对齐、训练均值库与谱线对照图
├─ test.py                           # 独立测试集推理调度、迁移样本跳过与 used_runs 汇总
├─ cli.py                            # infer CLI：test
└─ __main__.py                       # 支持 python -m raman.infer
```

### 2.8 分析层

```text
raman/analysis/
├─ pipeline.py                       # 分析调度：构建上下文并切换 single_model / level_only / cascade
├─ tasks.py                          # 分析任务与 DataLoader 构建
├─ level.py                          # 单层/单模型分析执行：IG、Grad-CAM、UMAP、SE
├─ aggregate.py                      # parent 子模型聚合分析
├─ ig.py                             # Integrated Gradients：输入通道重要性与类别波段重要性
├─ gradcam.py                        # Layer Grad-CAM：层级/分组重要性分析
├─ embedding.py                      # embedding 收集与 train + test 联合 UMAP 可视化
└─ se.py                             # 读取训练期 SE sidecar 并输出 SEBlock 缩放统计
```

### 2.9 Notebook 与数据目录

```text
colab/
└─ colab_unified.ipynb               # Colab 一体化 notebook：解压库、数据处理、训练、评估、分析、打包
notebooks/
├─ Cosmic_Ray_and_Baseline_Correction.ipynb # 单谱宇宙射线去除与 baseline 对照入口
├─ augmentation_effect_viewer.ipynb         # 在线增强效果查看
└─ preprocess_viewer.py                     # 预处理对照 notebook 的绘图与处理 helper
dataset/
└─ <数据集名>/
   ├─ init/                          # 原始归档解包后的初始数据
   ├─ init.npz                       # init 的压缩归档
   ├─ train/                         # 训练用清洗后数据
   ├─ init_test/                     # 手动放入的独立测试原始数据
   ├─ test/                          # 从 init_test 清洗生成的独立测试集
   ├─ audit_full_scan/               # full 分阶段审核报告、候选表和复核图
   ├─ audit_bad_band/                # 系统性坏段扫描输出
   ├─ delete/                        # 人工确认移除后的原始文件
   ├─ fig_init/                      # raman.shift 审图、delta.txt、delta_log.txt 与可选快照
   └─ fig_train/                     # 训练集均值图与高层级均值图
```

## 3. 记号说明

- $n$：目标样本索引，常用于表示当前正在分析的第 $n$ 条光谱
- $m$：临时样本索引，常用于在同一组或参考组内做求和、取中位数等统计
- $i$：位置索引，表示光谱序列中的第 $i$ 个波数采样点
- $L$：单条光谱的采样点数量
- $N$：当前目标文件夹或当前 batch 中的样本数量，具体含义由上下文决定
- $M$：参考组中的样本数量
- $c$：类别索引，表示第 $c$ 类
- $k$：PCA主成分个数
- $K$：类别总数
- $t$：训练轮次（epoch）或时间步索引
- $x$：输入光谱、输入特征或中间表示
- $y$：真实标签
- $z_n$：第 $n$ 条经过标准化后的光谱
- $d_n$：第 $n$ 条光谱相对组内中心谱的残差
- $\sigma_i$：第 $i$ 个波数采样点上的组内鲁棒尺度
- $\tau_z$：逐点 robust z-score 的判定阈值

### 3.1 函数命名约定

- `build_*`：构造对象、结构或中间配置
- `resolve_*`：把名称、路径、层级解析成确定结果
- `load_*`：从磁盘或实验目录读取配置、模型、元数据
- `save_*`：落盘保存模型、统计、图表或文本
- `compute_*`：纯数值计算，返回指标、权重、归因等结果
- `collect_*`：批量收集模型输出、embedding、统计数据
- `plot_*`：绘图并写出图片
- `run_*`：顶层流程或完整执行器
- `format_*`：把结果整理成文本或展示格式
- `_xxx`：模块内部 helper；如果只是无语义转发，优先删除或并回调用处

## 4. 离线数据预处理

### 4.1 常用命令

```
python -m raman.data pack <数据集名或profile>       # 打包init数据集
python -m raman.data unpack <数据集名或profile>     # 还原init数据集
python -m raman.data count <数据集名或profile>      # 统计数据集
python -m raman.data train <数据集名或profile>      # 从init直接清洗、按类别前缀合并并生成train
python -m raman.data test <数据集名或profile>       # 从init_test清洗生成独立测试集test
python -m raman.data plot <数据集名或profile>       # 从已有train独立生成fig_train审图
```

### 4.2 参数修改

离线清洗参数在 `raman/data/build.py` 里修改

清洗参数集成为 `DEFAULT_PIPELINE_CONFIG`

设置涵盖：

- 波段裁剪范围 `cut_min` / `cut_max`
- 统一参考波数轴点数 `target_points`
- 坏波段范围 `bad_bands`
- 基线校正方法 `baseline_method`
- 基线平滑参数 `baseline_lam`
- AsLS 不对称权重 `baseline_asls_p`
- 基线迭代次数 `baseline_max_iter`
- 基线拟合缓冲范围 `baseline_fit_min` / `baseline_fit_max`
- 宇宙射线去除阈值与启用数据集
- 训练集最小样本数 `min_samples_per_class`
- PCA 异常值过滤相关参数

不同数据集的目录名在 `raman/data/profiles.py` 里维护

训练集审图参数单独放在 `raman/data/plot.py` 的 `TrainPlotConfig` 中。`norm_method` 未指定时，会跟随当前运行配置中的标准化方法。

### 4.3 数据集目录结构

- `init/`：原始按测量文件夹组织的数据
- `init.npz`：`init/` 的打包版本
- `train/`：训练集离线清洗结果
- `init_test/`：手动放入的独立测试原始数据
- `test/`：由 `python -m raman.data test` 从 `init_test/` 清洗生成的独立测试集
- `fig_init/`：`raman.shift preview`、`plot-shift` 输出和 delta 记录
- `fig_train/`：训练集均值谱图与高层级均值谱图
- `audit_full_scan/`：分阶段审核报告、候选表和复核图
- `audit_bad_band/`：系统性坏段扫描输出
- `delete/`：人工确认移除后的原始 `.arc_data` 文件
- `pca_log.txt`：训练集 PCA 异常值剔除日志
- `cosmic_ray_removal_log.txt`：启用宇宙射线清理的数据集对应的替换统计日志

### 4.4 数据分析流程

离线阶段建议按以下顺序处理：

1. 准备或还原原始数据到 `init/`
2. 先用 `python -m raman.shift preview <数据集名或profile>` 查看原始 raw 中位谱总览，重点对比同属同前缀不同文件夹的谱形和峰位是否整体对齐
3. 如果发现同前缀不同文件夹存在系统性峰位偏移，先用 `raman.shift apply` 做平移修正，并用 `raman.shift plot-shift` 复核，再进入后续异常谱审核
4. 使用 `raman.audit full --stage ...` 对 `init/` 中的原始库做分阶段扫描，必要时先用 `raman.audit bad-band` 扫描系统性坏段
5. 结合审核报告和复核图人工确认异常谱
6. 使用 `raman.audit move` 将确认移除的原始 `.arc_data` 从 `init/` 移到 `delete/`
7. 使用 `python -m raman.data train <数据集名>` 从当前 `init` 或 `init.npz` 直接生成最终 `train/`
8. 按需使用 `python -m raman.data plot <数据集名>` 从已有 `train/` 生成或刷新 `fig_train/`
9. 使用 `python -m raman.data test <数据集名>` 对独立测试集生成一致的 `test/`

审核命令的输入仍然是 `init/` 中的原始文件，但评分前会临时执行当前离线预处理流程，包括宇宙射线去除、基线校正、裁剪、坏段剔除和统一波数轴插值

### 4.5 原始谱与平移审核

先用 `raman.shift preview` 看 `init/` 里的原始谱形态：

```bash
python -m raman.shift preview <数据集名或profile>
```

读取 `init/` 中的谱，按“属名 + 文件夹前缀”分组，输出到 `dataset/<数据集名>/fig_init/`。每张图有两个视图：

- 上半部分：raw 中位谱，用于直接观察原数据形态和整体峰位
- 下半部分：扣除 baseline 后按当前 `norm_method` 标准化的中位谱，并纵向错位排列，用于辅助比较谱形

同一属下同前缀小文件夹，主要峰位应该大体对齐。如果某个文件夹整体向左或向右偏移，应该先做平移审核，而不是直接交给异常谱审核

平移修正和复核使用：

```bash
python -m raman.shift apply <数据集名或profile> --folder <属名>/<小文件夹> --delta <平移量> --note "<备注>"
python -m raman.shift plot-shift <数据集名或profile> --folder <属名>/<小文件夹>
```

`--delta` 的单位是 `cm^-1`，正值向右平移，负值向左平移。当前累计平移量记录在 `fig_init/delta.txt`；每次平移动作会追加到 `fig_init/delta_log.txt`，包括单次增量、累计增量、修改文件数和可选备注

`plot-shift` 会输出单个文件夹平移前后的 raw 中位谱对照图，方便复核

默认 preview 会忽略以 `t` 结尾的迁移目录，例如 `KP06t`。需要把这些插入训练集的独立测试来源目录也纳入对照时，运行：

```bash
python -m raman.shift preview <数据集名或profile> --include-transferred
```

此时如果存在 `fig_init/delta_cs.txt`，程序会使用其中保存的迁移目录累计平移快照，避免后续原目录 delta 变化影响迁移目录审图。

平移审核完成后，再运行 `raman.audit full --stage ...` 做分阶段异常谱扫描

这样可以避免把系统性峰位偏移误判成类内相似性离群，也能让类内相似性阶段的同前缀参考池更可靠

如果 preview 中已经能看到明显坏段或整体下凹区域，可以先运行 `raman.audit bad-band` 做只读坏段扫描；如果主要问题是少数单谱无效或类内离群，则进入 `raman.audit full` 的分阶段流程

### 4.6 原始库审核与人工移除

当前审核入口已经收敛为三个命令：

```bash
python -m raman.audit full <数据集名或profile> --stage invalid
python -m raman.audit bad-band <扫描目标>
python -m raman.audit move <数据集名或profile> --from-list <delete_candidates.csv>
```

其中：

- `full`：主审核流程，按阶段扫描 `init/` 中的原始光谱
- `bad-band`：可选的系统性下凹坏段只读扫描，不移动文件
- `move`：把人工确认或阶段扫描确认的候选从 `init/` 移到 `delete/`

单谱无效和类内参考组异常都放进 `full --stage ...` 的阶段流程里处理

建议按“先排除明显无效，再做类内相似性”的顺序执行：

```bash
python -m raman.audit full <数据集名或profile> --stage invalid --max-spectrum-figures 200
python -m raman.audit full <数据集名或profile> --stage invalid --move --max-spectrum-figures 200

python -m raman.audit full <数据集名或profile> --stage class-similarity --max-spectrum-figures 200
python -m raman.audit full <数据集名或profile> --stage class-similarity --move --max-spectrum-figures 200
```

实际操作时可以先不加 `--move` 只看报告和图

确认 `delete_candidates.csv` 没问题后，再用同一阶段加 `--move` 执行移动

每移动一批后，建议再跑下一阶段。这样后续阶段的参考池不会被前一阶段已经确认的异常谱污染

#### 两个主审核阶段

##### invalid

只判断单谱自身是否已经失去有效光谱形态

主要证据包括：

- `invalid_missing_region`：原始波数覆盖不足，有长段缺失
- `invalid_flat_region`：连续平坦区过长，或全谱过多点贴近中位数
- `invalid_noise`：高频粗糙度高，同时平滑结构相对弱

##### class-similarity

用同属同前缀参考池判断单谱是否偏离本类

参考池数量不足时不自动判定

主要证据包括：

- `corr_ref`：目标谱和同前缀参考中位谱的相关系数
- `nearest_ref_corr`：目标谱和参考池中最相近单谱的最高相关系数
- `rmse_to_ref`：目标谱和参考中位谱的 RMSE
- `local_pos_*`：目标谱相对参考池的局部正残差异常
- `folder_candidate_count` / `folder_candidate_fraction`：同一小文件夹内候选是否过度集中

类内相似性阶段有一个保护逻辑：如果同一小文件夹里候选数量或比例过高，程序倾向于把它标成 `review_candidate`，并输出文件夹级复核图，而不是直接把整批都当作单谱离群删除

这类情况通常需要判断是整批采集偏移、标签/前缀问题，还是确实混入异常谱

#### 阶段输出

每次 `full` 会输出到带阶段和时间戳的目录：

```text
dataset/<数据集名>/audit_full_scan/<时间戳>_<stage>_<dry_run|move>/
```

输出文件包括：

- `summary.md`：中文总结报告，列出强异常移除候选、仅复核候选和阶段统计
- `summary.json`：记录数据目录、阶段、阈值配置、候选数量和是否执行移动
- `delete_candidates.csv`：只包含强异常 `remove_candidate`
- `review_candidates.csv`：只包含边界复核候选，不自动移动
- `delete_manifest.txt`：一行一个建议移除路径，便于人工快速浏览
- `all_spectra_scores.csv`：所有光谱的审核评分，便于后续排序筛选
- `figures/delete/`：删除候选复核图
- `figures/review/`：复核候选复核图
- `figures/folder_review/`：类内相似性阶段中文件夹集中命中时的文件夹均值对照图

`decision` 字段有四种常见取值：

- `keep`：当前阶段未发现需要处理的问题
- `review_candidate`：建议看图确认，但 `full --move` 不会自动移动
- `remove_candidate`：强删除候选，`full --move` 会移动这部分
- `skip`：预处理失败或输入不可用，原因写在 `reasons`

#### 系统性坏段扫描

如果怀疑某一段波数在大量样本中系统性下凹，可以先跑只读坏段扫描：

```bash
python -m raman.audit bad-band <数据集名或profile>
python -m raman.audit bad-band dataset/50种菌cos/init/Klebsiella/KAE03
python -m raman.audit bad-band dataset/50种菌cos/init --no-plot
```

`bad-band` 的命令行只保留扫描目标和是否画图；扫描区间、窗口宽度、抽样数量等参数在 `raman/audit/bad_band.py` 的配置里修改

输出目录默认为：

```text
dataset/<数据集名>/audit_bad_band/<时间戳>/
```

主要文件：

- `summary.md`：候选坏段摘要和判读说明
- `best_bad_band.csv`：当前配置下最合适的坏段区间
- `bad_band_overview.png`：候选坏段复核图，使用 `--no-plot` 时不生成

坏段扫描不会修改 `COMMON_BAD_BANDS` 或任何原始文件；是否把候选波段加入离线坏段配置，仍需要结合复核图人工判断

#### 审核后移动

`full --move` 会把当前阶段的 `delete_candidates.csv` 移到对应分类目录：

```text
dataset/<数据集名>/delete/Invalid Spectrum/
dataset/<数据集名>/delete/Class_Similarity_Outliers/
```

也可以先 dry-run，再用 `move` 手动执行：

```bash
python -m raman.audit move <数据集名或profile> --from-list dataset/<数据集名>/audit_full_scan/<时间戳>_<stage>_dry_run/delete_candidates.csv --dry-run
python -m raman.audit move <数据集名或profile> --from-list dataset/<数据集名>/audit_full_scan/<时间戳>_<stage>_dry_run/delete_candidates.csv
python -m raman.audit move <数据集名或profile> Burkholderia/BCC01/CELL8_Area01_000_shift.arc_data --reason Class_Similarity_Outliers --category Class_Similarity_Outliers --dry-run
```

移动规则为：

- 源路径必须来自 `dataset/<数据集名>/init/<属名>/<小文件夹>/...`
- 目标路径位于 `dataset/<数据集名>/delete/` 下；带 `--category` 时会进入对应分类目录
- 默认不覆盖已有文件
- CSV 清单优先使用 `reason_labels` 和 `delete_category` 列
- TXT 清单没有原因列，必须额外传 `--reason`
- 手动移动整属目录默认被拒绝，确实需要时才使用 `--allow-genus`

阶段阈值和删除分类统一在 `raman/audit/config.py` 的 `AuditConfig` 中维护。

### 4.7 构建训练集

从 `init/` 或 `init.npz` 直接生成最终训练目录 `train/`

由于采集数据按日期划分，原始数据集一般命名为 `类别+数字`

处理逻辑：

- 扫描 `init/` 或 `init.npz`

- 每个叶子小文件夹内部先独立执行宇宙射线去除、基线校正、裁剪、坏段删除和统一波数轴插值

- 按 `letters_sign` 规则提取类别前缀，把同一前缀的小文件夹合并成最终训练类别

- 合并时会给文件名加上原小文件夹前缀，避免不同小文件夹内的同名光谱互相覆盖

- 合并后的同一训练类别再按 PCA 重构误差做异常值过滤

每条光谱执行：

1. 读取 `.arc_data`
2. 单谱宇宙射线去除
3. 先在 `baseline_fit_min` 到 `baseline_fit_max` 范围内做基线校正，默认保留 400-2000 `cm^-1` 缓冲区
4. 基线扣除后，再裁剪到 `cut_min` 到 `cut_max`，默认 600-1800 `cm^-1`
5. 在裁剪后的原始波数轴上删除坏段
6. 对统一后的波数轴做线性插值，不跨坏段补点
7. 对同一分组样本按 PCA 重构误差做异常值过滤

如果某个分组预处理后样本数少于 `min_samples_per_class`，该分组会跳过

被 PCA 剔除的样本会记录到 `pca_log.txt`

`raman.data train` 只负责生成 `train/` 和日志。成功重建后会清空旧 `fig_train/`，避免旧图继续对应已经变化的数据。

需要审图时单独运行：

```bash
python -m raman.data plot <数据集名或profile>
```

绘图命令只读取已有 `train/`，不会刷新清洗结果。它会安全重建 `fig_train/` 中的类别均值图、层级均值图和层级汇总长图。

### 4.8 离线处理顺序细节

#### 宇宙射线去除

宇宙射线在拉曼光谱中通常表现为相对局部背景突然抬高的正向异常峰

这类异常和真实拉曼峰的差异在于：

- 真实峰一般具有相对稳定的物理峰宽和连续峰形
- 宇宙射线更容易表现为窄尖峰，或窄到中等宽度的突起片段
- 宇宙射线通常是正向异常，不作为负向异常处理

宇宙射线清理发生在坏段删除之前，并且默认不使用坏段 mask 参与检测，坏段只在绘图遮挡、基线估计和后续裁切插值阶段使用。这样可以避免 890-950 `cm^-1` 坏段边界附近的宇宙射线因为 mask 断开而无法被修复

对于单条光谱 $x \in \mathbb{R}^L$，$i$ 表示光谱序列中的第 $i$ 个波数采样点

先在位置 $i$ 附近取局部窗口 $\mathcal{W}_{\mathrm{cosmic\_ray}}(i)$，用窗口内中值估计该位置的局部正常强度：

```math
\tilde{x}_i^{(\mathrm{cosmic\_ray})}
=
\operatorname{median}
\left\{
x_{\ell}
\mid
\ell \in \mathcal{W}_{\mathrm{cosmic\_ray}}(i)
\right\}
```

- $\mathcal{W}_{\mathrm{cosmic\_ray}}(i)$ 表示以位置 $i$ 为中心的宇宙射线检测局部窗口
- $\ell$ 是窗口内的临时位置索引
- $\mathcal{W}_{\mathrm{cosmic\_ray}}(i)$ 的长度对应 `cosmic_ray_window_points`

局部残差定义为：

```math
r_i^{(\mathrm{cosmic\_ray})}
=
x_i
-
\tilde{x}_i^{(\mathrm{cosmic\_ray})}
```

为了避免局部强峰和少量异常点影响尺度估计，对残差使用 MAD 估计鲁棒尺度

```math
\sigma_{\mathrm{cosmic\_ray}}
=
1.4826 \cdot
\operatorname{median}_{i}
\left(
\left|
r_i^{(\mathrm{cosmic\_ray})}
-
\operatorname{median}_{\ell}
\left(
r_{\ell}^{(\mathrm{cosmic\_ray})}
\right)
\right|
\right)
```

对应的鲁棒 z-score 可写为

```math
z_i^{(\mathrm{cosmic\_ray})}
=
\frac{
r_i^{(\mathrm{cosmic\_ray})}
-
\operatorname{median}_{\ell}
\left(
r_{\ell}^{(\mathrm{cosmic\_ray})}
\right)
}
{\sigma_{\mathrm{cosmic\_ray}}}
```

只检测正向异常点，若某个位置满足

```math
z_i^{(\mathrm{cosmic\_ray})}
>
\lambda_{\mathrm{cosmic\_ray}}
```

则认为该位置是单谱内的疑似宇宙射线尖峰，并用局部中值替换

```math
x_i \leftarrow \tilde{x}_i^{(\mathrm{cosmic\_ray})}
```

其中 $\lambda_{\mathrm{cosmic\_ray}}$ 对应 `cosmic_ray_threshold`

#### AsLS 基线校正原理

AsLS（Asymmetric Least Squares，非对称最小二乘）是一种常用的光谱基线估计方法，其核心思想是通过“平滑约束 + 非对称加权”拟合一条平滑背景曲线，并抑制信号峰对基线估计的干扰

包含两个关键机制：

- 使用二阶差分项约束基线的平滑性
- 使用非对称权重减弱峰上方数据点对拟合结果的影响

设原始光谱为 $x \in \mathbb{R}^L$，基线为 $b \in \mathbb{R}^L$，则 AsLS 的优化目标可写为

```math
\min_{b} \sum_{i=1}^{L} w_i (x_i - b_i)^2 + \lambda \sum_{i=1}^{L-2} (b_{i+2} - 2b_{i+1} + b_i)^2
```

第一项是拟合误差，第二项衡量基线的局部弯曲程度

矩阵形式为

```math
\min_{b} (x - b)^T W (x - b) + \lambda b^T D^T D b
```

其中：

- $W = \mathrm{diag}(w_1, w_2, \dots, w_L)$为对角权重矩阵

  ```python
  matrix_w = sparse.diags(weights, 0)
  ```

- $D$为二阶差分矩阵，用于惩罚基线的“弯曲度”，保证基线平滑

  ```math
  D=
  \begin{bmatrix}
  1 & -2 & 1 & 0 & 0 & \cdots & 0 \\
  0 & 1 & -2 & 1 & 0 & \cdots & 0 \\
  0 & 0 & 1 & -2 & 1 & \cdots & 0 \\
  \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
  0 & 0 & 0 & \cdots & 1 & -2 & 1
  \end{bmatrix}
  \in \mathbb{R}^{(L-2)\times L}
  ```

  ```python
  D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
  ```

- $\lambda$ 为平滑参数，控制基线光滑程度，越大越平滑

在权重固定时，对目标函数关于 $b$ 求导并令其为零，可得到线性方程组

```math
(W + \lambda D^T D) b = W x
```

```python
matrix_b = (matrix_w + lam * (D.T @ D)).tocsc()     # W + λD^T D
baseline = spsolve(matrix_b, weights * spectrum)    # 解 b
```

由于权重 $w_i$ 本身依赖于当前基线估计 $b$，因此 AsLS 需要通过迭代方式求解

权重更新规则：

```math
w_i =
\begin{cases}
p, & x_i > b_i \\
1-p, & x_i \le b_i
\end{cases}
```

- 当 $x_i > b_i$ 时，该点更可能位于峰上方，赋予较小权重
- 当 $x_i \le b_i$ 时，该点更可能属于背景区域，赋予较大权重
- $p$ 控制“把高于基线的点当成峰并降低其影响”的强度，增大会使得基线更高，更容易被峰拉上去

```python
weights = np.where(spectrum > baseline, p, 1 - p)
```

所以 AsLS 的求解过程其实就是不断重复两步：

1. 固定当前权重，求解基线
2. 根据新的基线更新权重

经过多次迭代后，基线会逐步贴近背景区域，同时避开主要信号峰

#### airPLS 基线校正原理

airPLS（Adaptive Iteratively Reweighted Penalized Least Squares，自适应迭代重加权惩罚最小二乘）和 AsLS 一样，也是在“拟合误差 + 平滑惩罚”的框架下估计基线

它的核心区别是：airPLS 不再手动指定 AsLS 中的非对称参数 $p$，而是根据当前残差自动更新权重

仍记原始光谱为 $x$，基线为 $b$，二阶差分矩阵为 $D$，则每一轮固定权重时仍然求解：

```math
(W^{(q)} + \lambda D^T D)b^{(q)} = W^{(q)}x
```

其中 $q$ 表示迭代轮次

求得当前基线后，计算残差：

```math
d_i^{(q)} = x_i - b_i^{(q)}
```

对应的权重更新可写为：

```math
w_i^{(q+1)}
=
\begin{cases}
0, & d_i^{(q)} \ge 0 \\
\exp\left(
q \cdot \dfrac{|d_i^{(q)}|}
{\sum_{\ell:d_\ell^{(q)}<0}|d_\ell^{(q)}|}
\right), & d_i^{(q)} < 0
\end{cases}
```

airPLS 会让负残差点在下一轮中承担更高权重，而把正残差峰点的权重压低

当负残差总量已经很小，或者达到最大迭代次数时，迭代停止

和 AsLS 相比，airPLS 少了手动调节 $p$ 的步骤，参数主要集中在：

- $\lambda$：基线平滑强度，对应 `baseline_lam`
- 迭代次数：对应 `baseline_max_iter`

AsLS vs airPLS / arPLS

| 方法       | 核心思想                                                 | 主要参数     | 优点                         | 典型问题                                   |
| ---------- | -------------------------------------------------------- | ------------ | ---------------------------- | ------------------------------------------ |
| **AsLS**   | 用固定非对称权重压低峰的影响                             | $\lambda, p$ | 简单、快、可控               | 需要调 $p$，噪声大时容易低估基线           |
| **airPLS** | 自适应更新权重，去掉手动 (p)                             | $\lambda$    | 比 AsLS 自动，少一个参数     | 对噪声和权重指数更新较敏感，也可能低估基线 |
| **arPLS**  | 用负残差估计噪声水平，再用 logistic 权重平衡基线附近噪声 | $\lambda$    | 通常更稳健，尤其有加性噪声时 | 仍需调 $\lambda$，不是所有场景必胜         |

#### PCA 异常值过滤

PCA（Principal Component Analysis，主成分分析）是一种经典的线性降维方法，其基本思想是在原始高维空间中寻找一组新的两两正交的坐标轴，使样本在这些方向上的投影方差依次最大

通过保留前几个主成分，可以在尽量保留主要信息的同时去除冗余噪声

对于某一类别内的光谱数据矩阵记其样本矩阵为

```math
X \in \mathbb{R}^{N \times L}
```

首先对数据按列中心化，得到

```math
\tilde{X} = X - \mathbf{1}\mu^T
```

在中心化数据基础上，构造协方差矩阵

```math
S = \frac{1}{N}\tilde{X}^{T}\tilde{X}
```

对协方差矩阵进行特征分解，可得

```math
S = P \Lambda P^{T}
```

- $P$ 为特征向量矩阵，其列向量对应主成分方向
- $\Lambda$ 为对角特征值矩阵，其对角元素表示各主成分所对应的方差大小

样本在主成分空间中的投影（得分矩阵）为

```math
T = \tilde{X} P
```

在异常值剔除阶段，仅保留前 $k$ 个主成分来表征该类别的主要结构

```math
P_k \in \mathbb{R}^{L \times k}
```

则对应的低维表示为

```math
T_{k} = \tilde X P_{k}
```

基于前$k$个主成分，可将样本重构回原始空间，得到重构结果

```math
\hat{X} = T_k P_k^T + \mathbf{1}\mu^T
```

对该类别中的第 $n$ 个样本，使用重构均方误差作为异常度量

```math
e^{(n)} = \frac{1}{L} \sum_{i=1}^{L} \left(x_i^{(n)} - \hat{x}_i^{(n)}\right)^2
```

重构误差越大，说明该样本越难被该类别的主成分子空间表示，因此越可能是异常样本、噪声样本或标签不一致样本

为了实现类别内异常值剔除，对每个类别分别统计其样本重构误差分布，并采用分位数方式确定阈值

若设异常值剔除比例为$r$，则阈值定义为

```math
\tau = Q_{1-r}(e)
```

其中：

- $r$：异常值比例
- $Q_{1-r}(e)$：误差向量 $e$ 的 `1-r` 分位数

## 5. 训练数据输入

### 5.1 训练数据处理流程

离线阶段完成以后，`train/` 和 `test/` 中保存的是“已经完成基线校正、坏段剔除、统一波数轴对齐”的单条光谱文本文件

这些单条光谱进一步转换成模型真正使用的输入张量:

1. 从 `train/` 扫描目录树
2. 自动构建层级标签树和类别映射
3. 读取单条 `.arc_data`
4. 按当前训练配置构造多通道输入
5. 把单条样本堆叠成 batch
6. 模型最终接收 `[B, C, L]` 张量

`RamanDataset` 在扫描目录时会自动整理出：

- 训练层级：`level_1 ... level_N`
- 每层的类别映射 `label -> id` / `id -> label` 
- 上下层级关系 `parent_to_children`
- 每个样本对应的多层级标签编码
- 供切分/分组使用的 `leaf` 内部标识

这样训练、评估、分析和预测都不需要手工维护类别表，而是直接依赖目录结构得到统一的层级语义

### 5.2 在线预处理与增强

当前实现将增强划分为两类：

- Stage A：RAW 域增强，用于模拟更接近采集过程或仪器层面的扰动；
- Stage B：标准化后增强，用于模拟幅度较小的局部形状变化

#### Stage A：RAW 域增强

RAW 域增强发生在标准化之前，直接作用于原始强度光谱，主要用于模拟以下扰动：

- 仪器噪声
- 批次差异
- baseline 残留
- 波数轴标定误差

RAW 域会独立抽样这几类增强：

- `piecewise_gain`：分段缩放峰高，模拟不同波段相对峰高比例变化

- `noise`：高斯/泊松噪声

  ```math
\sigma(x) = a + b|x|
  ```
  
  在每个波数点上按“固定底噪 + 与峰高成比例的噪声”去扰动，更接近真实光谱采集时的噪声形态

- `baseline`：残余背景扰动

  分为强弱两类：

  **弱 baseline 扰动**：模拟幅度较小、形状较平滑的背景残留

  使用“线性趋势 + 低频正弦项”的组合来构造背景

  ```math
  b_{\text{weak}}(i) = \alpha i + \beta \sin(2\pi f i + \phi)
  ```

  **强 baseline 扰动**：模拟更明显的批次差异或仪器背景漂移

  不再限制为“线性 + 正弦”的简单形式，通过少量控制点在整条谱上构造一条分段平滑变化的低频曲线

  先在谱轴上随机放置若干个标记点，横坐标等距，纵坐标从均匀分布采样，再按样本幅值缩放

  ```python
  n_knots = np.random.randint(n_knots_min, n_knots_max + 1)
  xs = np.linspace(0, length - 1, n_knots, dtype=np.float32)
  ys = np.random.uniform(-1.0, 1.0, n_knots).astype(np.float32)
  ys *= np.random.uniform(amp_min, amp_max) * amp
  ```

  最后通过插值得到整条背景曲线

  ```math
  b_{\text{strong}}(i) = \mathrm{Interp}\big((i_1,y_1),\dots,(i_m,y_m)\big)
  ```

- `axis_warp`：模拟轻微的非刚性波数轴偏移

  先为原始坐标构造一个扰动后的坐标映射 $i' = i + \Delta(i)$

  $\Delta (i)$不是常数，随位置变化，所以不同波数位置的偏移量不同

  形变由两部分组成：一个线性项加一个正弦项

  ```math
  \Delta(i)=\alpha\,(i-i_0)+\beta \sin\!\left(2\pi \frac{i}{L}+\phi\right)
  ```

  把原始光谱视为定义在扰动坐标上的信号，再插值回原始规则网格

  相当于坐标扭曲后再重采样

  ```math
  \tilde{x}(i)=x\!\bigl(i'(i)\bigr)
  ```

增强先按各自概率独立抽样，再随机打乱顺序，最后只执行前 `max_pre_augs` 个

#### Stage B：标准化后增强

主通道在完成标准化后，还可以进一步叠加较弱的形状扰动

当前标准化后增强保留以下三类操作：

- `shift`：轻微峰位漂移 $\tilde{x}(i) = x(i-\Delta i)$

  让整条光谱沿着离散采样轴左右移动若干个点 

- `broadening`：峰展宽

  通过小尺度高斯卷积，让谱峰更圆滑、更宽

- `mask_attenuate`：局部缺失或污染

  随机选取一个局部区间，把那一段信号按比例压低，做平滑的局部衰减

  区间中间保持完全衰减，边缘 20% 用余弦 ramp 平滑过渡

  ```python
  ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
  window[:edge] = ramp
  window[-edge:] = ramp[::-1]
  ```

最终实际执行操作数受 `max_post_augs` 限制

### 5.3 标准化方法

在线输入中的 `Normalize(...)` 对应 `raman.data.input.normalize_spectrum`

标准化是逐条光谱独立完成的，不使用训练集全局均值或全局方差

这样做的目的，是让模型更关注单条谱内部的相对峰形，而不是被采集强度、曝光、样本浓度或整体背景幅度主导

当前训练默认使用 `norm_method = "snv"`，如果需要切换到最大最小归一化，可以把配置中的 `norm_method` 改为 `"minmax"`

#### SNV 标准化

SNV（Standard Normal Variate）对单条光谱自身做均值-标准差标准化

设离线清洗后、进入在线输入前的单条光谱为

```math
x = (x_1,\dots,x_L)
```

先计算该条谱自己的均值和标准差：

```math
\mu_x
=
\frac{1}{L}
\sum_{i=1}^{L}
x_i
```

```math
\sigma_x
=
\sqrt{
\frac{1}{L}
\sum_{i=1}^{L}
\left(
x_i-\mu_x
\right)^2
}
```

SNV 后第 $i$ 个采样点为：

```math
\hat{x}_i
=
\frac{
x_i-\mu_x
}{
\max(\sigma_x,\epsilon)
}
```

其中 $\epsilon$ 是很小的数值下限，用来避免平坦谱或近似常数谱导致除零

SNV 的含义是：把每条谱平移到均值 0，并缩放到标准差约为 1，它会消除单条谱整体强度尺度和整体偏移的影响，更适合让模型学习相对峰形、峰间比例和局部形态差异

需要注意的是，SNV 是逐条谱内部标准化，因此它不会保留“这条谱整体比另一条谱强多少”这种绝对强度信息

如果绝对强度本身有判别意义，SNV 会削弱这部分信息

#### 最大最小归一化

最大最小归一化（Min-Max Normalize）把单条光谱线性缩放到固定范围，当前实现对应 $[0,1]$

对单条光谱先计算：

```math
x_{\min}
=
\min_{i=1,\dots,L} x_i,
\qquad
x_{\max}
=
\max_{i=1,\dots,L} x_i
```

归一化后第 $i$ 个采样点为：

```math
\hat{x}_i
=
\frac{
x_i-x_{\min}
}{
\max(x_{\max}-x_{\min},\epsilon)
}
```

最大最小归一化的含义是：保留单条谱内部点与点之间的相对高低关系，并把最小值压到 0、最大值拉到 1

它比 SNV 更直观，输出范围固定，适合需要非负输入范围的场景

它的代价是对极端点更敏感：如果一条谱里残留尖峰或异常高点，最大值会被异常点占据，其它正常峰会被整体压低

因此在当前流程中，若使用 `minmax`，更依赖前面的宇宙射线去除和异常谱审核质量

### 5.4 模型输入

标准化后的单通道光谱不会直接送进模型，而是会按配置构造成多通道输入

所有通道共享同一个 RAW 增强后的光谱 $x_{\mathrm{raw}} \in \mathbb{R}^{L}$，再基于这条共享谱线构造各个输入通道

当前各通道的语义如下：

- `base`：$x_{\mathrm{base}} = \mathrm{PostAug}\!\left(\mathrm{Normalize}(x_{\mathrm{raw}})\right)$

- `smooth`：$x_{\mathrm{smooth}} = \mathrm{Normalize}\!\left(\mathrm{SG}(x_{\mathrm{raw}})\right)$

- `d1`：$x_{\mathrm{d1}} = \mathrm{Normalize}\!\left(\mathrm{D1}(\mathrm{SG}(x_{\mathrm{raw}}))\right)$

  一阶导本身对局部噪声非常敏感，如果不先平滑，导数通道会更容易被高频噪声主导

```math
x = \mathrm{Stack}\!\left(x_{\mathrm{base}},\,x_{\mathrm{smooth}},\,x_{\mathrm{d1}}\right)
```

#### SG平滑

SG（Savitzky–Golay）平滑本质是：在滑动窗口内做局部多项式最小二乘拟合，再用拟合多项式在中心点的值替代原始值

在每个位置 $i$，取一个窗口$[i-m, \dots, i, \dots, i+m]$

在这个窗口内，用一个低阶多项式去拟合数据：
```math
x(i + k) \approx a_0 + a_1 k + a_2 k^2 + \cdots + a_d k^d
```

其中：

- $i$：当前中心点
- $k$：相对偏移，$k \in [-m, m]$
- $d$：多项式阶数

在窗口内求解：

```math
\min_{a_0,\dots,a_d}\sum_{k=-m}^{m}\left(x(i+k)-\sum_{j=0}^{d}a_jk^j\right)^2
```

进一步地，将窗口内数据写成向量形式：

```math
\mathbf{x}_i=
\begin{bmatrix}
x(i-m)\\
x(i-m+1)\\
\vdots\\
x(i+m)
\end{bmatrix}
```

构造多项式拟合的设计矩阵：

```math
A=
\begin{bmatrix}
1 & (-m) & (-m)^2 & \cdots & (-m)^d\\
1 & (-m+1) & (-m+1)^2 & \cdots & (-m+1)^d\\
\vdots & \vdots & \vdots & & \vdots\\
1 & m & m^2 & \cdots & m^d
\end{bmatrix}
```

则局部最小二乘问题可写为：

```math
\min_{\mathbf{a}}\|\mathbf{x}_i-A\mathbf{a}\|_2^2
```

对$a$求导，其解析解为：

```math
\hat{\mathbf{a}}=(A^TA)^{-1}A^T\mathbf{x}_i
```

由于平滑后的输出为中心点 $k=0$ 处的函数值，其实只需要关注$a_0$

这等价于用一组固定系数对窗口内数据做线性加权
```math
\mathbf{h}=
\begin{bmatrix}
1 & 0 & \cdots & 0
\end{bmatrix}
(A^{T}A)^{-1}A^{T}
```

```math
\tilde{x}(i)=\sum_{k=-m}^{m}h_k\,x(i+k)
```

$h_k$ 为 SG 平滑对应的卷积系数，仅由窗口大小 $2m+1$ 和多项式阶数 $d$ 决定，与具体输入信号无关

SG 平滑本质上可以看作一种由局部多项式最小二乘推导得到的固定卷积核线性滤波

### 5.5 训练集、验证集、测试集

- 训练入口使用的基础数据目录是 `train/`
- 训练集与验证集都从 `train/` 内部划分得到
- 如果实验目录下已有 `train_files.json` 和 `test_files.json`，会优先复用原切分
- 如果没有，就按 `split_level` 重新分组切分

当前训练代码中：

- `train_dataset = RamanDataset(..., augment=True)`，用于训练
- `test_dataset = RamanDataset(..., augment=False)`，用于训练过程中的验证

这里的 `test_dataset` 只是训练阶段的验证集视图，并不等同于外部独立测试集

真正的独立测试集位于 `test/`，不参与训练期切分

## 6. 模型

```
输入 x [B, C, L]
        │
        ▼
┌─────────────────────────────┐
│        多尺度 Stem           │
│  Conv(k=3) ┐                │
│  Conv(k=7) ├─ concat → 64ch │
│  Conv(k=15)┘                │
└────────────┬────────────────┘
             ▼
         AvgPool (↓2)
             ▼
┌──────────────────────────────┐
│         CNN Backbone         │
│   (ResNeXt + SE Bottleneck)  │
│                              │
│ Layer1: 64   ×2 blocks       │
│ Layer2: 128  ×2 blocks + ↓2  │
│ Layer3: 256  ×2 blocks + ↓2  │
│ Layer4: 384  ×2 blocks + ↓2  │
└────────────┬───────────────-─┘
             ▼
        1×1 Conv (384 → D)
             ▼
     [B, D, L] → [B, L, D]
             ▼
┌──────────────────────────────┐
│ Transformer + Positional Enc │
└────────────┬─────────────────┘
             ▼
┌──────────────────────────────┐
│       Stat Pooling           │
│        mean + std → 2D       │
└────────────┬─────────────────┘
             ▼
┌──────────────────────────────┐
│      Cosine Classifier       │
└────────────┬────────────────-┘
             ▼
        logits [B, num_classes]
```

### 6.1 总体结构

当前模型定义在 `raman/model.py`，整体可概括为：

```text
输入 [B, C, L]
→ backbone（cnn / identity）
→ encoder（transformer / lstm / none）
→ pooling（attn / stat）
→ classifier（cosine / linear）
```

其中：

- `B` 表示 batch size
- `C` 表示输入通道数，由输入构造配置决定
- `L` 表示离线统一后的光谱长度

这套结构将局部峰形提取、跨波段关系建模和最终分类解耦，便于围绕 backbone、序列编码器、池化方式和分类头进行消融与替换

### 6.2 Backbone

前端特征提取器由 `backbone_type` 控制：

- `cnn`：使用 1D CNN 主干提取局部峰形特征
- `identity`：跳过 CNN，仅做平均下采样和 `1×1` 通道投影

当前主线配置为例说明 `cnn` 主干，其整体流程为：

```
输入 [B, C, L]
→ stem branches
→ concat
→ AvgPool1d(/2)
→ layer1
→ AvgPool1d(/2) + layer2
→ AvgPool1d(/2) + layer3
→ AvgPool1d(/2) + layer4
→ 1×1 Conv projection
```

```python
def _forward_feat_extractor(self, x):
    """把输入 `[B, C, L]` 转成 backbone 序列特征"""
    if self.cnn_backbone_on:
        x = torch.cat([branch(x) for branch in self.stem_branches], dim=1)
        x = self.stem_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.proj(x)
```

对应的通道变化为：

1. stem 输出 `64` 通道，并先做一次 `AvgPool1d(kernel_size=2)`
2. `layer1`：`64 -> 64`
3. `layer2`：先池化再进入残差块，`64 -> 128`
4. `layer3`：先池化再进入残差块，`128 -> 256`
5. `layer4`：先池化再进入残差块，`256 -> 384`
6. 最后通过 `1×1 Conv` 投影到统一的 `proj_dim`（当前主线常用 `192`）

CNN 主干并不在残差块内部用 stride 做降采样，而是把长度压缩显式放在 stem 之后以及 layer 开始之前

把“尺度压缩”和“卷积特征变换”解耦

主干网络在这个过程中压缩4次，降为 $L/16$ 

#### 6.2.1 多尺度 stem

由 `stem_kernel_sizes` 控制分支数与卷积核尺度

模型会先根据分支数把总通道数分配给各分支，再为每个分支构建一个独立的卷积块

```python
self.stem_branches = nn.ModuleList(
    [
        make_conv_block(
            self.in_channels,
            branch_channels,
            kernel_size=kernel_size,
            make_activation=self.make_backbone_activation,
        )
        for kernel_size, branch_channels in zip(
            kernel_sizes,
            self._split_channels(self.stem_out_channels, len(kernel_sizes)),
        )
    ]
)
```

在通道维拼接后再统一做一次平均池化

允许模型在输入端同时观察不同尺度的局部模式：

- 小卷积核更容易捕捉尖峰、窄峰和局部突变
- 较大卷积核更适合感知宽峰、缓变背景和峰包络

对于拉曼光谱，这比固定单一尺度更符合不同峰宽与局部形态并存的特点

#### 6.2.2 残差 bottleneck 块

CNN 主干的基本单元是 `ResidualBottleneck1D`

每个 block 包含：

1. `1x1 Conv` 降维  
2. `3×1 Conv1d` 主卷积变换
3. `1x1 Conv` 升维  
4. `SEBlock1D` 通道重标定
5. residual shortcut 与输出激活

可写为：

```math
x \rightarrow \mathrm{Conv1d}(\text{kernel}=1) \rightarrow \mathrm{Conv1d}(\text{kernel}=3) \rightarrow \mathrm{Conv1d}(\text{kernel}=1) \rightarrow \mathrm{SE} \rightarrow +\;\mathrm{shortcut} \rightarrow \phi
```

若输入输出通道数不同，shortcut 会自动使用 `1×1 Conv + BN` 做投影

```python
self.conv_reduce = make_conv_block(
    in_channels,
    mid_channels,
    kernel_size=1,
    make_activation=make_activation,
    padding=0,
)
self.conv_mid = make_conv_block(
    mid_channels,
    mid_channels,
    kernel_size=3,
    make_activation=make_activation,
    groups=groups,
)
self.conv_expand = make_conv_block(
    mid_channels,
    out_channels,
    kernel_size=1,
    padding=0,
)
```

整体输出形式为：

```math
x_{\text{out}}=\phi(\mathrm{SE}(F(x))+\mathrm{shortcut}(x))
```

```python
def forward(self, x):
    """执行 bottleneck 主分支和 shortcut 残差融合"""
    residual = self.shortcut(x)
    out = self.conv_reduce(x)
    out = self.conv_mid(out)
    out = self.conv_expand(out)
    out = self.se(out)
    return self.out_act(out + residual)
```

#### 6.2.3 mid_channels

中间 bottleneck 宽度由 `resolve_mid_channels()` 根据 block 类型决定

```python
"""按 ResNet/ResNeXt 配置计算 bottleneck 中间通道数"""
block_type = str(block_type).lower()
if block_type == "resnext":
    mid_channels = int(out_channels * (base_width / 64.0)) * cardinality
    return max(mid_channels, int(cardinality))
if block_type == "resnet":
    return max(int(out_channels // bottleneck_ratio), 1)
```

**ResNet 模式**

```math
\text{mid\_channels} = \max \left(\left\lfloor \frac{\text{out\_channels}}{\text{bottleneck\_ratio}} \right\rfloor, 1\right)
```

默认 `bottleneck_ratio=4` ，若某一 stage 的输出通道为 `128`，则其中间宽度为 `32`

**ResNeXt 模式**

```math
\text{mid\_channels} =
\max\left(
\underbrace{\text{out\_channels} \cdot \frac{\text{base\_width}}{64}}_{\text{每组的宽度}} \times \underbrace{\text{cardinality}}_{\text{组数}},
\text{cardinality}
\right)
```

- cardinality（最重要）：卷积多分支，分组

- base_width：控制每个分支内部有多粗

默认主线用 `cardinality=4`、`base_width=4`，中间 `3x1 Conv1d` 会分组计算，而不是全部通道一起进行卷积

结构一样，但每个 group 独立学习，权重完全不同

这意味着 block 会先在多个子空间内提取局部模式，再在输出端统一融合

```math
x=\bigl[x^{(1)},x^{(2)},\dots,x^{(G)}\bigr]\\
F(x)=\mathrm{Concat}\!\left(\mathrm{Conv}_1(x^{(1)}),\mathrm{Conv}_2(x^{(2)}),\dots,\mathrm{Conv}_G(x^{(G)})\right)
```

对峰形模式较多、局部结构复杂的拉曼光谱，这种分组表达通常比单一路径卷积更灵活

#### 6.2.4 SE 模块

每个 bottleneck 后面都可以接 `SEBlock1D`，由 `se_use` 控制

SE 的过程为：

1. 对特征图做全局平均池化，得到通道描述向量
2. 通过两层全连接层生成通道权重
3. 将权重乘回原特征图，实现通道重标定

输入特征为 $x \in \mathbb{R}^{B \times C \times L}$

对每个通道，在长度维上做全局平均池化：

```math
z_{b,c}=\frac{1}{L}\sum_{i=1}^{L}x_{b,c,i}
```

压缩成一个标量，作为这个通道的全局描述，也就是判断这个通道整体响应强不强

```python
self.pool = nn.AdaptiveAvgPool1d(1)
```

得到的通道描述向量输入一个小型 MLP，输出每个通道的权重(0-1)

```math
\mathrm{scale} = \sigma\left(W_2 \, \delta(W_1 z)\right)
```

```python
self.fc = nn.Sequential(
    nn.Linear(channels, hidden_channels),
    make_activation(inplace=False),
    nn.Linear(hidden_channels, channels),
    nn.Sigmoid(), # 限制在(0,1)
)
```

通常中间会先降维再升维

```math
\mathbb{R}^{C} \rightarrow \mathbb{R}^{C/r} \rightarrow \mathbb{R}^{C}
```

主要有两个原因：

1. 减少参数量
2. 增加非线性建模能力

如果直接学一个$C\rightarrow C$ 的全连接，也能做，但参数更多，SE 采用瓶颈结构更经济

得到权重后再乘回原特征图

```math
\tilde{x}_{b,c,i} = \mathrm{scale}_{b,c}  \odot x_{b,c,i}
```

```python
def _compute_scale(self, x):
    """根据当前 batch 的通道全局响应计算缩放系数"""
    batch_size, channels, _ = x.size()
    scale = self.pool(x).view(batch_size, channels)
    return self.fc(scale)
def forward(self, x):
    batch_size, channels, length = x.size()
    scale = self._compute_scale(x)
    scale = scale.unsqueeze(-1).expand(batch_size, channels, length)
    return x * scale
```

相当于对通道做抑制或增强操作

### 6.3 Encoder

backbone 投影并转置后的序列特征记为 $x \in \mathbb{R}^{B \times L \times C}$，再交给序列编码器处理

```python
feat = self._forward_feat_extractor(x)
feat = feat.permute(0, 2, 1)
feat = self._forward_sequence_encoder(feat)
```

`encoder_type` 支持三种模式：

- `transformer`
- `lstm`
- `none`

在 backbone 已经提炼出局部响应之后，encoder 进一步对序列表示建模不同波段之间的上下文关系

对拉曼光谱来说，可以理解为：

- backbone 更像“先找出哪些局部峰形有响应”
- encoder 更像“让峰 A 感知峰 B 是否同时出现，以及这些峰之间的组合关系”

#### transformer

在进入 Transformer 之前，模型会先加上一层一维正余弦位置编码 `PositionalEncoding1D`

当前使用的是标准正余弦位置编码，最大支持长度为 `1000`

```python
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
```

对位置 `pos` 和通道维度 `2i / 2i+1`，编码形式为：
```math
PE(pos, 2i) =\sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)\\
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
```

前向时直接把位置编码和序列特征相加：

```math
x \leftarrow x + PE
```

对于拉曼光谱，同样的局部形状如果出现在不同波数位置，含义可能完全不同，因此位置信息不能丢

利用 PyTorch 的 `nn.TransformerEncoderLayer` 再外包一层 `nn.TransformerEncoder`

单层结构可以概括成两部分：

1. 多头自注意力
2. 前馈网络 FFN

自注意力的核心形式是：

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

在多头机制下，输入会被投影到多个子空间分别做注意力，然后再拼接回来

FFN 则作用在每个位置上，对已经融合上下文的信息再做非线性变换

启用了 `norm_first=True`，也就是先做 LayerNorm，再进入注意力和 FFN 子层

**Post-LayerNorm**

```
x
 │
 ├── Sublayer(x)   (Attention 或 FFN)
 │
 ├── + residual (x)
 │
 └── LayerNorm
     │
     y
```

对应可写为：

```math
x_1 = \mathrm{LayerNorm}(x+\mathrm{Attention}(x))\\
x_2 = \mathrm{LayerNorm}(x_1+\mathrm{FFN}(x_1))
```

**Pre-LayerNorm**

```
x
 │
 ├── LayerNorm
 │
 ├── Sublayer
 │
 ├── + residual
 │
 └── y
```

对应可写为：
```math
x_1 =x+\mathrm{Attention}( \mathrm{LayerNorm}(x))\\
x_2 = x_1+\mathrm{FFN}(\mathrm{LayerNorm}(x_1))
```

在工程上，Pre-LayerNorm 的主要优点是残差路径更直接，训练通常更稳定，尤其在层数增加或训练条件不够理想时更明显

```python
self.pos_encoder = PositionalEncoding1D(d_model=self.proj_dim)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=self.proj_dim,
    nhead=self.config.transformer_nhead,
    dim_feedforward=self.config.transformer_ffn_dim,
    dropout=self.config.transformer_dropout,
    batch_first=True,
    activation="gelu",
    norm_first=True,
)
self.transformer = nn.TransformerEncoder(
    encoder_layer,
    num_layers=self.config.transformer_layers,
)
```

#### lstm

当 `encoder_type="lstm"` 时，序列编码器会切换成 `nn.LSTM`

它和 Transformer 不同，不是靠自注意力一次性看全局，而是按序列顺序逐步更新隐藏状态，因此更接近经典时序建模方式

```python
self.lstm = nn.LSTM(
    input_size=self.proj_dim,
    hidden_size=self.lstm_hidden,
    num_layers=self.lstm_layers,
    dropout=self.lstm_dropout if self.lstm_layers > 1 else 0.0,
    bidirectional=self.lstm_bidirectional,
    batch_first=True,
)
```

输入维度固定为前端投影后的 `proj_dim`，也就是当前默认的 `192`

如果打开双向 LSTM，那么最终序列维度会变成：
```math
\text{seq\_dim} = 2 \times \text{lstm\_hidden}
```

否则就是:

```math
\text{seq\_dim} = \text{lstm\_hidden}
```

LSTM 的核心是通过门控机制控制信息的保留、遗忘和输出，从而缓解普通 RNN 在长序列上容易出现的梯度消失问题

它包含三类门：

- 输入门（input gate）
- 遗忘门（forget gate）
- 输出门（output gate）

门控输出都取决于新的输入和上一时刻隐状态输出

对时刻 `t`

<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png" style="zoom:33%;" />

<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png" style="zoom:33%;" />

<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png" style="zoom:33%;" />

<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png" style="zoom: 33%;" />

直觉上：

- 遗忘门决定旧记忆保留多少
- 输入门决定新信息写入多少
- 输出门决定当前时刻暴露多少隐藏状态

如果把 Transformer 理解成“显式建模任意两个位置之间的关系”，那么 LSTM 更像“沿着波数轴逐步积累上下文信息”

但它的局限也很明确：

- 远距离位置之间的关系传播需要经过多步递推
- 并行性不如 Transformer
- 对“峰 A 和很远处峰 B 的直接组合关系”建模不如自注意力直接

因此，在本项目里 LSTM 更适合作为一个可对照的序列编码器基线

#### none

当 `encoder_type="none"` 时，模型会跳过序列编码器，直接把 backbone 投影后的序列特征送入后续 pooling

这相当于保留前端局部峰形提取能力，但不额外引入 Transformer 或 LSTM 去建模跨位置上下文关系

此时 `_forward_sequence_encoder()` 会直接返回输入本身

### 6.4 Pooling

encoder 输出的是序列特征：
```math
x \in \mathbb{R}^{B \times L \times C}
```

需要将序列特征在长度维上压缩

`pooling_type` 支持：

- `attn`：注意力池化
  
  模型会先为每个位置学习一个打分，再在长度维上做 softmax，最后对序列特征加权求和
  
  ```math
  \mathrm{score}_i=f_{\mathrm{att}}(x_i) \qquad 
  \mathrm{attn}_i=\frac{\exp(\mathrm{score}_i)}{\sum_{j=1}^{L}\exp(\mathrm{score}_j)} \\
  
  x=\sum_{i=1}^{L} \mathrm{attn}_i\,x_i
  ```
  
  使用一个小型 MLP 对每个位置的特征打分
  
  ```python
  self.att_pool = nn.Sequential(
      nn.Linear(self.seq_dim, self.seq_dim // 2),
      nn.GELU(),
      nn.Dropout(att_pool_dropout),
      nn.Linear(self.seq_dim // 2, 1),
  )
  attn = torch.softmax(self.att_pool(x), dim=1)
  feat = (x * attn).sum(dim=1)
  ```
  
  更依赖数据规模与正则化设置，在样本较少时更容易过拟合
  
- `stat`：统计池化
  
  模型不会显式学习位置权重，而是直接对序列维做统计汇聚
  
  使用均值与标准差拼接
  
  ```math
  \mu = \frac{1}{L} \sum_{i=1}^{L} x_i \qquad 
  \sigma = \sqrt{\frac{1}{L} \sum_{i=1}^{L}(x_i - \mu)^2}\\
  x = \mathrm{Concat}(\mu, \sigma)
  ```
  
  这种方式不需要额外学习位置打分，因此更稳定，也更不容易因为样本量有限而过拟合

经过pool后输出的特征尺度为`[B, C]`，`C`的大小受池化头方式影响

### 6.5 Classifier

分类头由 `cosine_head` 控制：

```python
if self.cosine_head:
    self.head = CosineClassifier(
        self.feat_dim,
        self.num_classes,
        scale=self.cosine_scale,
    )
else:
    self.head = nn.Linear(self.feat_dim, self.num_classes)
```

- `True`：`CosineClassifier`

  输入特征`[B, C]`和类别向量`[K, C]`都会先做 L2 归一化
  ```math
  \hat{x} = \frac{x}{\|x\|_2} \qquad \hat{w}_k = \frac{w_k}{\|w_k\|_2}
  ```

  这样内积就对应余弦相似度

  ```math
  \cos (\theta_k) = \hat x \cdot \hat w_k
  ```

  因此第 $k$ 类的 logit 可写为

  ```math
  z_k = s \cdot \cos(\theta_k)
  ```

  $s$ 为余弦头缩放参数，因为余弦相似度本身范围被压得很窄，直接送进 softmax输出概率不会特别尖锐，增大 $s$ 会让不同类别之间的 softmax 概率差异更明显，从而增强交叉熵的监督信号

  不再利用类别权重向量的模长，而主要根据特征与类别原型方向是否一致来做判别

- `False`：`Linear`
  ```math
  z = Wx + b
  ```
  
  同时利用两种信息
  
  1. 特征和类别权重的方向是否一致
  2. 特征向量本身的模长大小
  
  普通 `Linear` 的表达更自由，但分类边界也更容易同时受特征方向与范数共同影响

## 7. 训练

### 7.1 训练入口与实验目录

训练统一从根目录的 `train.py` 进入

当前入口层可显式修改的主要内容包括：

- `CURRENT_TRAIN_LEVEL`  必须显式提供
- `TRAIN_ONLY_PARENT_NAME`
- `TRAIN_ONLY_PARENT`  索引优先级高于名称
- `OVERRIDE_ALIGN_LOSS_WEIGHT`
- `OVERRIDE_SUPCON_TAU` 
- `OVERRIDE_SUPCON_LOSS_WEIGHT`
- `OVERRIDE_OUTPUT_DIR`

若没有显式指定 `output_dir`，运行时会自动生成实验目录

```
output/<数据集名>/<时间戳>/
```

实验目录内会保存：

```
<EXP_DIR>/
├─ config.yaml
├─ hierarchy_meta.json
├─ class_names.json
├─ train_files.json
├─ test_files.json
├─ logs/
│  ├─ run.log
│  ├─ config.txt
│  └─ <model_tag>.log
├─ level_1/
│  ├─ level_1_model.pt
│  └─ level_1_model.se_stats.pt
└─ level_2/
   ├─ level_2_parent_3_model.pt
   └─ level_2_parent_3_model.se_stats.pt
```

`hierarchy_meta.json` 是后续预测、评估和分析都会复用的关键元数据文件，记录

- 层级顺序
- 每层类别名
- `parent_to_children`
- 本次训练得到的全局模型和各 parent 子模型的相对路径
- 哪些父类因为只有一个子类而被直接记录为“确定映射”

`*.se_stats.pt` 是与模型同目录、同前缀保存的 SEBlock 统计 sidecar，仅在启用 SE 且验证阶段产生统计时生成

### 7.2 层级训练逻辑

训练入口里设置的是 `CURRENT_TRAIN_LEVEL`

- 数据集完整层级始终由 `train/` 的目录树自动扫描得到
- `CURRENT_TRAIN_LEVEL` 只表示“这次训练实际要训练的那一层”

当 `train_per_parent=True` 时，训练行为是：

- 顶层没有父层，因此训练全局模型
- 若当前层存在父层，则按父类拆成多个子任务
- 若某个父类下只有一个子类，则不训练该 parent 子模型，只在层级元数据中记录这条确定关系

如果当前实验目录缺少上一级模型或单子类记录，训练开始时会打印提示，提醒先训练哪一级

若给定 `TRAIN_ONLY_PARENT`，则直接按父类索引限制训练范围

### 7.3 训练/验证切分

训练代码扫描的是 `train/`，然后在内部再做 train/val 切分

切分逻辑位于 `raman/training/split.py`：

1. 根据 `split_level`、`train_split` 和 `seed` 生成一份切分
2. 再检查当前实验目录中是否已经存在 `train_files.json` 和 `test_files.json`
3. 若存在，则直接复用已有切分
4. 若不存在，则把新生成的切分写入实验目录

默认配置：

- `split_level = "leaf"`
- `train_split = 0.8`
- `seed = 42`

切分文件在实验目录中生成，后续继续在同一实验目录下训练更细层级时，会优先复用同一套 `train_files.json / test_files.json`，主要目的是：

- 顶层模型和子模型尽量使用同一套训练/验证划分基准
- 多次补训练时，验证结果具有可比较性
- 不会因为每次重切分而造成实验波动

### 7.4 训练优化

主要涵盖四类机制：

1. 参数更新侧：`Adam + weight_decay + CosineAnnealingLR`
2. 结构分区侧：不同模块使用不同学习率
3. 数据吞吐侧：`DataLoader` 预取与并行加载
4. 验证选择侧：用综合早停分数而不是单独看 `TestLoss`

#### 参数更新侧

默认主学习率 `learning_rate = 4e-4`

学习率调度参数为：

- `scheduler_Tmax = epochs`
- `scheduler_eta_min = 1e-5`

学习率受到两层作用：

- `Adam` 负责根据梯度及其历史统计更新参数 $\theta_t$
- `CosineAnnealingLR` 控制全局学习率 $ \eta_t$ 沿余弦曲线逐步退火

```math
\theta_{t+1}=\theta_t-\eta_t \cdot \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\\
\eta_t=\eta_{\min}+\frac{1}{2}\left(\eta_{\max}-\eta_{\min}\right)\left(1+\cos\left(\frac{\pi t}{T_{\max}}\right)\right)
```

```python
optimizer = optim.Adam(
    param_groups,
    weight_decay=1e-4,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.scheduler_Tmax,
    eta_min=config.scheduler_eta_min,
)
```

调度器直接作用在带分组学习率的 optimizer 上

#### 分组学习率

当前训练器不是对全模型使用统一学习率，而是按模块分组：

- 输入 stem：`0.6 × learning_rate`
- backbone 其他部分：`1.0 × learning_rate`
- 分类头：`1.2 × learning_rate`

这是一种“结构分区优化”，含义是：

- 输入 stem 更接近底层峰形提取，学习率略低，避免训练前期把基础局部结构扰乱
- backbone 主体保持标准学习率，承担主要表征学习
- 分类头学习率略高，便于更快贴合当前层级的类别边界

#### DataLoader 设置

当前 DataLoader 相关默认配置来自 `config.py`：

- `train_loader_num_workers = 4`
- `eval_loader_num_workers = 4`
- `loader_pin_memory = True`
- `loader_persistent_workers = True`
- `loader_prefetch_factor = 2`

训练集 `DataLoader` 当前使用的是普通 `shuffle=True`

这部分配置当前主要影响的是训练效率

- 每个 epoch 的吞吐速度
- CPU 预取是否能跟上 GPU
- 验证阶段是否会被 I/O 明显拖慢
- batch 内正样本对的自然形成概率

最后这一点对 `SupCon Loss` 很重要，因为当前 `SupCon` 是在普通随机 batch 上计算的，它的有效性会受到 batch 内同类样本数量的影响

#### Early Stop

当前对模型的综合得分评价依赖整体准确率与宏平均 F1

```math
\text{score} = w_{f1} \cdot \text{MacroF1} + w_{acc} \cdot \text{Accuracy}
```

默认权重为：

- `early_stop_w_f1 = 0.6`
- `early_stop_w_acc = 0.4`

`Accuracy` 表示整体分类准确率，即所有样本中被正确分类的比例

```math
\mathrm{Accuracy} = \frac{\sum_{c=1}^{K} TP_c}{N}
```

- $TP_c$ 表示第 $c$ 类的真正例数

反映的是整体正确率，因此更容易受到头部类别样本数的影响

`MacroF1` 表示先分别计算每个类别的 F1，再在类别维度做简单平均

定义精确率和召回率

```math
P_c = \frac{TP_c}{TP_c + FP_c} \qquad  R_c = \frac{TP_c}{TP_c + FN_c}
```

- $FP_c$ 表示被错误预测为第 $c$ 类的样本数
- $FN_c$ 表示真实为第 $c$ 类但未被预测正确的样本数

```
                           Predicted
                     ┌────────────┬────────────┐
                     │ Positive   │ Negative   │
┌────────────────────┼────────────┼────────────┤
│ Actual Positive    │ TP         │ FN         │
├────────────────────┼────────────┼────────────┤
│ Actual Negative    │ FP         │ TN         │
└────────────────────┴────────────┴────────────┘
```

```python
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
```

则该类的 F1 为

```math
F1_c = \frac{2 P_c R_c}{P_c + R_c}
```

最终的宏平均 F1

```math
\mathrm{MacroF1} = \frac{1}{K}\sum_{c=1}^{K} F1_c
```

对当前这种类间不平衡、不同类别难度差异又较大的拉曼任务来说，Macro F1 更能反映尾部类和难类是否真正学到

```python
score = (
    config.early_stop_w_f1 * macro_f1
    + config.early_stop_w_acc * test_acc
)
if score >= best_score:
    best_score = score
    torch.save(model.state_dict(), best_model_path)
```

另外：

- 训练日志中的 `TrainLoss(cls)` 是主分类损失的 batch 平均
- `AlignLossW` 和 `SupConLossW` 是已经乘过当前 epoch 权重后的辅助损失
- 验证日志里的 `TestLoss` 来自验证阶段单独计算的 `CrossEntropyLoss`

### 7.5 训练损失

训练时的总损失由三部分组成：

```math
L_{\text{total}}(t)=L_{\text{cls}}(t)+\lambda_{\text{align}}(t)L_{\text{align}}+\lambda_{\text{supcon}}(t)L_{\text{supcon}}
```

其中：

- `L_cls(t)`：当前 batch 的主分类损失
- `L_align`：当前层 embedding 的类内紧凑约束
- `L_supcon`：监督式对比损失

```python
loss_cls_each = criterion(logits_valid, y_valid)

if config.use_severity_weight:
    with torch.no_grad():
        prob = torch.softmax(logits_valid, dim=1)
        severity_w = _compute_severity_weights(prob, y_valid)
    loss_cls = (loss_cls_each * severity_w).mean()
else:
    loss_cls = loss_cls_each.mean()

loss_align = align_loss_fn(feat, hier_labels)
loss_supcon = supcon_loss_fn(feat, hier_labels)

loss_total = loss_cls + align_w * loss_align + supcon_w * loss_supcon
```

#### 主分类损失

由 `criterion(...)` 计算逐样本分类损失

```math
L_{\mathrm{cls\_each}}^{(n)}(t) = FL\left(p_t^{(n)}; w_{n}(t)\right)
```

- $p_{t}^{(n)}$ 表示第 $n$ 个样本在真实类别上的预测概率
- $w_{n}(t)$ 是第 $n$ 个样本所属类别在当前训练轮次 $t$ 使用的类别权重

如果启用了 `severity weight`，则主分类损失写为：

```math
L_{\mathrm{cls}}(t)=\frac{1}{N}\sum_{n=1}^{N} s_n\,L_{\mathrm{cls\_each}}^{(n)}(t)
```

若未启用 `severity weight`，则退化为普通均值：

```math
L_{\mathrm{cls}}(t)=\frac{1}{N}\sum_{n=1}^{N}L_{\text{cls\_each}}^{(n)}(t)
```

其中 $s_n$ 是第 $n$ 个样本的错误严重程度权重

##### Focal Loss

在光谱层级分类任务中，不同样本难度差异较大，容易样本会主导梯度，导致模型忽略难样本

Focal Loss 在交叉熵损失基础上增加调制因子，抑制易样本、强调难样本，从而把更多训练预算分配给当前还没学好的样本

`CrossEntropy Loss`

```math
CE(p_t) = - \log(p_t)
```

`Focal Loss`

```math
FL(p_t; w) = -  w (1 - p_t)^\gamma \log(p_t)
```

$\gamma$：控制对易样本的抑制程度

```python
criterion = FocalLoss(
    gamma=config.gamma,
    weight=base_class_weights,
    ignore_index=-1,
)
```

`FocalLoss` 是先基于未加权交叉熵计算 $p_t$ 和 focal 因子，再用当前 epoch 的类别权重对逐样本损失做重加权

```python
def forward(self, logits, targets):
    ce_loss = nn.functional.cross_entropy(
        logits,
        targets,
        weight=None,  # 不带权重
        reduction="none",
        ignore_index=self.ignore_index,
    )

    pt = torch.exp(-ce_loss)
    focal_factor = (1 - pt) ** self.gamma

    if self.weight is not None:
        sample_weight = self.weight[targets]
        loss = sample_weight * focal_factor * ce_loss
    else:
        loss = focal_factor * ce_loss

    return loss
```

返回的是逐样本 loss 向量，后续才能继续叠加 `severity weight`

##### 类别权重

在进入动态重加权之前，训练器会先根据当前训练层的标签分布构造基础类别权重 `base_class_weights`

当前使用对数平滑：

1. 统计当前训练层每个类别的样本数
2. 对计数做下界保护，避免出现 0
3. 按下式计算基础权重

   ```math
   \text{base\_class\_weight}_c = \frac{1}{\log(\text{count}_c + 1.5)}
   ```

4. 再归一化到平均值为 1

   ```math
   \text{base\_class\_weight}_c \leftarrow \frac{\text{base\_class\_weight}_c}{\frac{1}{K} \sum_{j=1}^{K} \text{base\_class\_weight}_j}
   ```

```python
counts = np.bincount(level_labels[valid], minlength=num_classes)
counts = np.maximum(counts, 1)
base_class_weights = 1.0 / np.log(counts + 1.5)
base_class_weights = base_class_weights / base_class_weights.mean()
```

在照顾少数类的同时避免极端长尾下权重过大，导致训练振荡

##### EMA 动态类别权重

在训练过程中，类别难度并不是固定的

某些类别虽然样本数不一定最少，但可能持续更难学，因此仅靠静态权重不够

当前实现会对每个类别维护一条基于 CrossEntropy 的 EMA 难度轨迹，并据此动态调整类别权重：

初始EMA设为1

```python
ema_class_ce = torch.ones(num_classes, device=device)
ema_alpha = 0.9
lambda_diff = 0.3
ema_start_epoch = 10
```

逻辑可以概括为：

1. 对每个类别统计当前 batch 内的平均 `CrossEntropy`

2. 用 EMA 平滑历史难度：

   ```math
   \mathrm{EMA}_c(t)=\alpha\,\mathrm{EMA}_c(t-1)+(1-\alpha)\,\mathrm{CE}_c^{\text{batch}}
   ```

   ```python
   ce_each = F.cross_entropy(logits_valid, y_valid, reduction="none")
   for c in range(num_classes):
       class_mask_c = (y_valid == c)
       if class_mask_c.any():
           mean_ce_c = ce_each[class_mask_c].mean()
           ema_class_ce[c] = ema_alpha * ema_class_ce[c] + (1.0 - ema_alpha) * mean_ce
   ```
   
3. 计算相对难度：

   ```math
   \mathrm{raw\_diff}_c=\frac{\mathrm{EMA}_c(t)}{\frac{1}{K}\sum_{j=1}^{K}\mathrm{EMA}_{j}(t)}
   ```

4. 用相对难度修正基础类别权重

   ```math
   \mathrm{ema\_class\_weight}_c
   =
   \mathrm{base\_class\_weight}_c
   \cdot
   \Bigl(1+\lambda(\mathrm{raw\_diff}_c-1)\Bigr)
   ```

5. 再归一化到平均值为 1

```python
if config.use_ema and epoch >= ema_start_epoch:
    raw_diff = ema_class_ce / (ema_class_ce.mean() + 1e-12)
    diff_factor = 1.0 + lambda_diff * (raw_diff - 1.0)
    ema_class_weights = base_class_weights * diff_factor
    ema_class_weights = ema_class_weights / (ema_class_weights.mean() + 1e-12)
    criterion.weight = ema_class_weights
```

##### 错误惩罚

除了类别层面的重加权，当前训练器还会在样本层面再做一次“错误严重程度”加权

- 如果模型没把真实类排到第一，但已经排到第二第三，说明它并不是完全错离谱
- 如果模型以很高置信度把样本分错，应当放大该样本在主损失中的贡献

当前实现里，会对每个有效样本：

1. 计算 `softmax(logits)`
2. 取前 `min(3, num_classes)` 个预测类别
3. 统计真实类别在 top-k 中的排名
4. 根据排名和 top-1 置信度给样本乘一个额外权重

当前规则按类别数自适应：

- 二分类：不额外使用 `severity weight`，权重为1

- 三分类：
  - 真实类排第二，样本权重为 `0.95`，如果高置信度错判则升到 `1.05`
  - 真实类排第三，样本权重为 `1.10`，如果高置信度错判则升到 `1.25`
- 四类及以上：
  - 真实类排第二，样本权重为 `0.90`，如果高置信度错判则升到 `1.1`
  - 真实类排第三，样本权重为 `1.00`，如果高置信度错判则升到 `1.35`
  - 真实类排第四及以后，样本权重为 `1.10`，如果高置信度错判则升到 `1.35`

其中高置信度阈值也按类别数调整：

- 三分类使用 `0.88`
- 四类及以上使用 `0.85`

##### 总结

用 Focal Loss 处理“难样本概率问题”

用 EMA 类别权重处理“持续难学类别问题”

用 severity weight 处理“错误结构严重程度问题”

#### Align Loss

实现的 `AlignLoss` 更接近一种 batch 内经验中心约束

1. 在当前 batch 中取出某一类别的全部有效样本 embedding
2. 用这些样本的均值作为该类别在当前 batch 内的经验中心
3. 计算该类别样本到这个经验中心的平方距离
4. 对当前 batch 内所有有效类别的类内方差做平均

设当前 batch 中类别 c 的样本索引集合为

```math
S_c=\{\, n \mid y_n = c \,\}
```

则该类别在当前 batch 内的经验中心为

```math
\mathrm{center}_c^{(\text{batch})}= \frac{1}{|S_c|}
\sum_{n \in S_c}  x_n
```

```python
center_c = feat_c.mean(dim=0, keepdim=True)
```

对应的类内紧凑项为

```math
L_c^{(\text{align})}=\frac{1}{|S_c|}\sum_{n \in S_c}\|x_n - \mathrm{center}_c^{(\text{batch})}\|_2^2
```

```python
diff_c = feat_c - center_c
radial_c = (diff_c * diff_c).sum(dim=1)
loss_sum += radial_c.mean()
```

对当前 batch 内所有有效类别取平均

```math
L_{\text{align}}
=
\frac{1}{|C_{\text{valid}}|}
\sum_{c \in C_{\text{valid}}}
L_c^{(\text{align})}
```

```python
return loss_sum / valid_group_count
```

其中：

- $C_{\text{valid}}$ 表示当前 batch 中样本数大于 1 的有效类别集合
- 如果当前 batch 没有任何可用类别，当前实现直接返回 `0`

#### SupCon Loss

实现的 `SupCon Loss` 是单视角监督式对比损失，不是双视图对比学习

`SupCon` 是直接在当前 batch 的特征表示上做同类拉近、异类推远

核心思想：

- 在特征空间中约束样本的相对距离
- 同类样本靠近
- 不同类样本远离
- 允许同一类别内部存在多个簇，因此它不强制类内单中心

可以拆成 5 步：

1. 对 embedding 做 L2 归一化

   ```math
   z_n = \frac{x_n}{\|x_n\|_2}
   ```

   ```python
   z = F.normalize(feat, p=2, dim=1)
   ```

2. 计算温度缩放后的两两相似度

   ```math
   \mathrm{sim}(n,m) = \frac{z_n^\top z_m}{\tau}
   ```

   ```python
   sim = torch.matmul(z, z.t()) / self.tau
   ```

3. 定义正样本集合：

   ```math
   P(n)=\{\,m \mid m\neq n,\; y_m = y_n\,\}
   ```

   ```python
   off_diag_mask = torch.ones_like(sim, dtype=torch.bool)
   off_diag_mask.fill_diagonal_(False)
   y = y.view(-1, 1)
   pos_mask = (y == y.t()) & off_diag_mask # 标出正样本位置
   ```

4. 对正样本的对数概率取平均，得到单个 anchor 的损失：

   ```math
   L_n=
   -\frac{1}{|P(n)|}
   \sum_{p\in P(n)}
   \log
   \frac{\exp(\mathrm{sim}(n,p))}
   {\sum_{m\neq n}\exp(\mathrm{sim}(n,m))}
   = 
   -\frac{1}{|P(n)|}
   \sum_{p\in P(n)}
   \left(
   \mathrm{sim}(n,p)-\log\sum_{m\neq n}\exp(\mathrm{sim}(n,m))
   \right)
   ```

   > 负样本在 $m\neq n$ 这里体现，如果某个异类样本和样本$n$的相似度很高，会把分母拉大，正样本对应的概率就会被压低，最终损失变大
   >
   > 分子显式拉近正样本，分母通过和“所有其他样本”竞争，隐式压低异类相似度

5. 最后，对所有有效 anchor 求平均

   ```math
   L_{\text{supcon}}
   =
   \frac{1}{|I_{\text{valid}}|}
   \sum_{n\in I_{\text{valid}}} L_n
   ```

其中：

- $I_{\text{valid}}$表示当前 batch 中至少有一个正样本对的 anchor 集合
- 如果整个 batch 没有任何正样本对，当前实现直接返回 `0`
- 当前 batch 来自普通随机 `shuffle`，没有专门为对比学习设计的 batch sampler
- 因此 `SupCon` 的有效性依赖于 batch 内是否自然出现足够多的同类样本对

### 7.6 训练损失之间的分工

- `FocalLoss` 解决“分错”和“难样本”
- `base_class_weights + ema_class_weights` 解决哪些类在当前训练阶段更需要额外关注
- `severity weight` 解决哪些错误更危险、更值得纠正
- `AlignLoss` 解决当前 batch 内的类内紧凑性
- `SupConLoss` 解决类间分离与类内相对距离结构

当前训练闭环可以概括为：

- 用 `FocalLoss` 学分类边界
- 用 `base_class_weights` 做静态类别平衡
- 用 `ema_class_weights` 在训练中后期按类别难度动态修正权重
- 用 `severity weight` 提高高置信错判样本的学习强度
- 用 `AlignLoss` 收紧当前层的类内分布
- 用 `SupConLoss` 拉开 embedding 的相对结构

## 8. 评估

### 8.1 测试集评估

深度模型评估入口在仓库根目录`evaluate.py`

需要手动设置这几个参数：

- `EXP_DIR`
- `EVAL_LEVEL`
- `INHERIT_MISSING_LEVELS`
- `EVAL_ONLY_LEVEL`
- `EVAL_ONLY_PARENT`

评估默认针对训练阶段保存下来的 test split，即实验目录中的 `test_files.json` 对应样本

保证评估时使用的类别空间、层级关系、模型组织方式和训练阶段保持一致，避免因为重新扫描目录或重新构造标签映射而引入额外偏差

如果实验目录里缺少 `train_files.json / test_files.json`，评估器会按训练配置中的 `split_level`、`train_split` 和 `seed` 重新构造 test split，再继续评估

### 8.2 评估流程

主流程可以概括为：

1. 从 `EXP_DIR` 加载 `config.yaml` 和 `hierarchy_meta.json`
2. 把 `dataset_root` 重新对齐到训练阶段目录，重建 `RamanDataset(augment=False)`
3. 解析目标评估层级 `eval_level`，得到从顶层到目标层的 `level_order`
4. 加载这条层级链路上需要的模型
5. 对每个测试样本逐层向下预测，直到目标层级
6. 把预测结果映射到当前评估的展示类别空间
7. 汇总准确率、Macro F1、Macro Recall、分类报告和混淆矩阵

对于按父类拆开的层，级联逻辑是：

- 先用上一级模型得到父类预测
- 再找到该父类对应的子模型
- 如果该父类只有一个子类且训练时没有单独保存模型，就直接继承这个唯一子类

这和真实预测流程保持一致，而不是把每一层都当成互不相关的独立分类任务

### 8.3 统计口径开关

`INHERIT_MISSING_LEVELS`这个开关控制的是：

- 如果样本在目标评估层级没有有效标签
- 是否回退到它实际存在的最深有效层级继续参与统计

当它为 `True` 时：

- 更适合看整体级联效果
- 对层级不完整、或者某些分支没有继续下钻的实验更友好
- 评估器会为这些样本构造一个兼容的展示类别空间

当它为 `False` 时：

- 只有目标层真实有标签的样本才参与统计
- 指标更严格，也更“纯”
- 更适合单独看“这一层本身到底分得怎么样”

更接近“目标层本身的纯分类能力”，而不是整条级联链路的兼容输出能力

开启继承后，报告里的类别顺序会出现“保留父类”和“子类”混合的显示类名

这是评估器为了兼容未下钻分支而构建的展示空间，不等同于简单读取某一层的全部原始类名

### 8.4 评估输出

`evaluate.py` 会在实验目录内生成：

```text
<EXP_DIR>/<EVAL_LEVEL>_test_result/
```

主要文件和用途如下：

- `test_eval_results.csv`：逐样本结果表，适合后续再做二次统计或错误样本回查
- `classification_report.txt`
  
  统一格式的分类报告，包含每类的 `precision / recall / f1-score / support`
  
  同时包含 `accuracy`、`macro avg` 和 `weighted avg`
- `confusion_matrix_raw.csv`：原始混淆矩阵计数表
- `confusion_matrix.png`：混淆矩阵热图，图中输出按行归一化的百分比，并同时标出原始计数

在终端会明确打印：

- `Accuracy`：整体判对率
- `Macro F1-score`：类别均衡视角下的综合分类表现
- `Macro Recall`：各类召回是否均衡

`classification_report.txt` 里的 `weighted avg` 更适合辅助判断头部类是否明显拉高了整体表现

### 8.5 PCA + SVM 基线

除了深度模型级联评估，仓库里还保留了一条传统机器学习基线作为对照：

- `pca_svm_baseline.py`

它和深度模型评估共用同一份训练/测试切分，因此可以做相对公平的对比

1. 从 `train/` 中按训练时的 split 提取 train/test 样本
2. 选择第一个输入通道，或把全部通道展平为静态特征向量
3. 对输入做 `StandardScaler`
4. 做 `PCA`
5. 用 `SVM` 训练并测试
6. 输出准确率、分类报告、混淆矩阵和 PCA 散点图

拉曼光谱本身是高维连续向量，不同波段的数值尺度和波动范围可能并不一致，先做 `StandardScaler`，把各维特征标准化到相近尺度

```python
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)
```

PCA 的作用是把原始高维光谱投影到若干主成分方向上，用更低维的表示保留数据中的主要变化趋势

PCA 会学习一组主方向$P_k$，并把样本映射到低维空间：

```math
T_k = X P_k
```

```python
pca = PCA(n_components=context.pca_n_components, random_state=context.random_state)
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.transform(x_test_std)
```

SVM 则作为经典的判别模型，基本思想是在特征空间中寻找一个分类间隔尽可能大的决策边界

对于线性可分的理想情形，其硬间隔形式可写为：

```math
\min_{w,b}\frac{1}{2}\|w\|_2^2
\quad
\text{s.t.}\quad
y_n(w^\top x_n+b)\ge 1
```

代码实际使用 `sklearn.svm.SVC`，对应的是更一般的软间隔 SVM，并允许结合核函数在隐式高维空间中进行非线性分类

```python
svm = SVC(C=context.svm_c, kernel=context.svm_kernel, gamma=context.svm_gamma)
svm.fit(x_train_pca, y_train)
y_pred = svm.predict(x_test_pca)
```

- `C`：软间隔 SVM 里的惩罚系数，默认使用1

- `kernel`：决定 SVM 是在线性空间分类，还是先把数据隐式映射到更高维后再分类，默认用`rbf`(高斯径向基核)

  ```math
  K(x_i,x_j)=\exp(-\gamma \|x_i-x_j\|_2^2)
  ```

- `gamma`：RBF 核里的关键参数，决定高斯核衰减得多快，默认`scale`

  ```math
  \gamma = \frac{1}{n_{\text{features}}\cdot \mathrm{Var}(X)}
  ```

  让 `gamma` 根据当前特征尺度自动调节

输出目录为：

```text
<EXP_DIR>/<LEVEL>_baseline_test_result/
```

最常用的结果文件有

- `metrics.txt`：记录 Accuracy、PCA 保留维数、解释方差比例和分类报告
- `confusion_matrix.png`：基线模型的混淆矩阵热图
- `pca_scatter.png`：训练集在 PCA 前两维上的散点图，适合快速观察类间可分性和不同类别的重叠程度

这条基线更接近“标准化后的静态光谱特征 + 传统分类器”的能力上限，而不是层级级联模型的直接替代物

### 8.6 独立测试集推理

独立测试原始谱先放在 `dataset/<数据集>/init_test/`，再执行：

```powershell
python -m raman.data test <数据集名或profile>
```

该命令会使用和训练集一致的物理预处理流程生成 `dataset/<数据集>/test/`，但不会执行训练集 PCA 异常值过滤。infer 只读取处理后的 `test/`。

推理入口统一放在 `raman.infer`：

```powershell
python -m raman.infer test --exp-dir "output/肠杆菌/五分类去除K/20260523_070649_92.1%" --level level_1
```

如果不想用命令行，可以直接改根目录 `infer_test.py` 顶部配置后运行：

```python
EXP_DIR = r"output/肠杆菌/五分类去除K/20260523_070649_92.1%"
LEVEL = "level_1"
FOLDER = None
```

`infer_test.py` 会从模型配置自动读取数据集，并默认使用 `dataset/<数据集>/test/`

输出目录按当前选择的模型 run 落在 `test_result/` 下；多层级级联场景会落在实验目录对应层级下。典型结构为：

```text
<结果根>/test_result/
├─ used_runs.json
├─ summary.txt
├─ skipped_transferred_samples.txt   # 仅跳过迁移样本时按需生成
└─ CS01KP/
   ├─ CS01KP_file.txt
   └─ spectra.png
```

`CS01KP_file.txt` 保留旧版文件夹推理的逐谱 top-k 格式，并在开头增加该文件夹的实际类别、多数投票结果、正确谱数和正确比例

`spectra.png` 默认显示该测试文件夹的所有测试谱和 `Test Mean`。启用 `--plot-train-mean` 后，会额外显示实际类别 `Train Mean`；如果多数投票误判，还会显示误判类别的训练均值线

`summary.txt` 逐文件夹汇总 `expected_label`、`expected_in_model`、`predicted_label`、`majority_count`、`total_count`、`correct_count`、`correct_ratio` 和 `folder_correct`

## 9. 分析

### 9.1 模式

分析入口保留在仓库根目录`analyze.py`

通过 `ANALYSIS_MODE` 选择三种模式：

- `single_model`：传入具体 `run_*` 或 `best/run_*` 目录，只分析一个模型
- `level_only`：传入实验根目录，只分析目标层；如果该层按 parent 拆模，会逐个加载并聚合解释结果
- `cascade`：传入实验根目录，按目标层的级联上下文解析实际使用的各层模型，并记录完整 `used_runs.json`

常用入口配置：

```python
ANALYSIS_MODE = "single_model"
RUN_DIR = ""
EXP_DIR = ""
TARGET_LEVEL = "level_1"
PARENT_IDX = None
HEATMAP_SEPARATE_CLASS_PLOTS = False
```

### 9.2 输出

分析会围绕当前模型或聚合任务输出多种解释结果，核心包括：

- 输入通道重要性 `channel_importance_IG.png`
- 各类别波段重要性热图 `band_importance_heatmap.png`
- 每类完整逐点波段重要性表 `band_importance_per_class.csv`
- 各层或各 stage 的重要性图 `layer_importance.png`
- embedding 可视化图

聚合分析的对应波段文件会增加 `_aggregate` 后缀，例如 `band_importance_heatmap_aggregate.png` 和 `band_importance_per_class_aggregate.csv`。

默认 `HEATMAP_SEPARATE_CLASS_PLOTS = False`，会输出一张类别汇总热图。改为 `True` 后，会改为按类别分别输出 `band_importance_heatmap__<类别路径>.png`；完整逐点 CSV 仍然保留。

这些结果并不在回答同一个问题：

- 输入通道重要性回答“模型更依赖哪一种输入表示”
- 波段热图回答“对某个类别而言，哪些波数位置最能驱动判别”
- 层重要性回答“模型决策更依赖前段特征提取，还是后段高层表征”
- embedding 图回答“不同类别在特征空间里是否真的被拉开”

### 9.3 Integrated Gradients（IG）

Integrated Gradients（IG）是一种输入归因方法，用来衡量输入各维度对目标输出的贡献

IG 的核心思想是：从一个参考输入逐步走到真实输入，在整条路径上累计梯度，从而得到更稳定的特征贡献估计

设输入张量为$x \in \mathbb{R}^{B \times C \times L}$，记$\mathrm{IG}_{n,c,i}$为第 $n$ 个样本、第 $c$ 个通道、第 $i$ 个波数位置的 Integrated Gradients 归因值

```math
\mathrm{IG}_i(x)
=
(x_i-x_i^{(0)})
\int_0^1
\frac{\partial F\!\left(x^{(0)}+\alpha(x-x^{(0)})\right)}{\partial x_i}
\, d\alpha
```

其中：

- $x$：真实输入
- $x^{(0)}$：参考输入(baseline)

- $\alpha \in [0,1]$：从baseline到真实输入的插值系数
- $F(\cdot)$：当前要解释的目标输出

IG 主要承担两类分析任务：

1. 输入通道重要性分析
2. 类别波段重要性分析

**baseline 的选择**

当前实现中，baseline 不是全零向量，而是从数据中估计得到的均值光谱

```python
baseline = _compute_baseline_mean_spectrum(loader, device, num_batches=num_batches)
```

最终得到形状为 `[1, C, L]` 的平均谱

**数值近似与目标函数**

由于积分一般无法直接解析求解，当前代码使用离散步长近似

```python
alphas = torch.linspace(0, 1, steps, device=device)
```

对每个离散步长 $\alpha_m$，先构造插值输入

```math
x^{(m)} = x^{(0)} + \alpha_m (x-x^{(0)})
```

再对目标输出求梯度，最后做平均：

```python
x_step = (b + alpha * (x - b)).detach().requires_grad_(True)
score = logits.gather(1, target.view(-1, 1)).sum()
score.backward()
total_grad += x_step.grad.detach()

avg_grad = total_grad / float(steps)
ig = (x - b) * avg_grad
```

当前实现采用的离散近似形式可以写成：

```math
\mathrm{IG}(x)
\approx
\bigl(x-x^{(0)}\bigr)
\odot
\frac{1}{M}
\sum_{m=1}^{M}
\nabla_x F\!\left(x^{(m)}\right)
```

其中 $M$ 是积分步数，对应代码中的 `steps`

IG 的输出张量形状为 `[B, C, L]`

先通过 `compute_ig_batches(...)` 统一计算原始 IG，再分别调用做不同层面的汇总

#### 输入通道重要性

对于输入通道重要性，IG 张量在 batch 维和长度维上取绝对值平均，得到每个输入通道的平均归因强度；再在通道维上归一化，得到相对贡献占比
$$
I_c
=
\frac{1}{BL}
\sum_{b=1}^{B}
\sum_{i=1}^{L}
\left|
\mathrm{IG}_{b,c,i}
\right|
$$

```python
channel_importance = ig.abs().mean(dim=(0, 2))
channel_importance = channel_importance / (channel_importance.sum() + 1e-8)
```

该结果更适合比较不同输入表示在整体判别中的相对重要性

#### 类别波段重要性

先沿通道维对归因绝对值取平均，得到形状为 `[B, L]` 的单样本波段归因：

```python
ig_band = ig.abs().mean(dim=1)
```

对应公式为：

```math
a_n(i)
=
\frac{1}{C}
\sum_{j=1}^{C}
\left|
\mathrm{IG}_{n,j,i}
\right|
```

对于类别 $c$，若其样本集合记为 $S_c$，则类别波段重要性可以写成：

```math
A_c(i)
=
\frac{1}{|S_c|}
\sum_{n \in S_c}
a_n(i)
=
\frac{1}{|S_c|}
\sum_{n \in S_c}
\left(
\frac{1}{C}
\sum_{j=1}^{C}
\left|
\mathrm{IG}_{n,j,i}
\right|
\right)
```

```
compute_class_band_importance_ig(...)
    ├─ compute_ig_batches(...)
    │    └─ 先算每个 batch 的原始 IG，保存为 [B, C, L]
    └─ compute_band_importance_from_ig(...)
         └─ 再把原始 IG 汇总成每类的波段重要性
```

### 9.4 Layer Grad-CAM

当前项目还会进一步分析模型内部不同层或不同 stage 的贡献

使用的是一种面向一维网络的 Layer Grad-CAM 风格分析，并不恢复输入位置上的热图，而是把每层的激活与梯度进一步压缩成单个重要性分数，用于比较不同层或不同 stage 的相对贡献

回答：

- 模型决策更依赖前段局部峰形提取，还是后段高层表征
- CNN、Transformer、LSTM 等不同模块中，哪些部分对当前输出更关键

如果某一层既有较强激活，又对目标类别的输出具有较高梯度敏感性，那么这层对当前决策通常更重要

因此，当前代码把该层的重要性定义为

```math
\mathrm{Score}=\mathrm{mean}\left(\left|A \odot G\right|\right)
```

其中：

- $A$ ：该层前向输出激活
- $G$ ：该层输出关于目标 logit 的梯度

会递归查找并保留：

- `conv1` 或 `input_proj`
- `ResidualBottleneck1D`
- `TransformerEncoderLayer`
- `LSTM`

按类型合并分组

在 `LayerGradCAMAnalyzer` 中，每个待分析层都会注册两类 hook：

- `activations`：缓存该层激活

  ```python
  def f_hook(module, inp, out):
      if isinstance(out, (tuple, list)):
          out = out[0]
      self.activations[name] = out.detach()
  ```

- `gradients`：缓存该层输出梯度

  ```python
  def b_hook(module, grad_in, grad_out):
      g = grad_out[0]
      self.gradients[name] = g.detach()
  ```

### 9.5 SE 模块缩放统计

如果模型启用了 `SEBlock1D`，训练期会在验证阶段累计每个 SEBlock 的缩放统计，并在最佳模型更新时保存为与模型同目录、同前缀的 sidecar

统计不保存每个 batch 的完整 scale，而是保存紧凑统计结果

- sample_count
- channel_mean
- channel_std
- channel_min
- channel_max

计算第$m$个模块的平均通道缩放向量

```math
s^{(m)}_j
=
\frac{1}{B}
\sum_{b=1}^{B}
s^{(m)}_{b,j}
```

再对这组通道缩放值做摘要统计

统计量的解释：

- `mean`：该 SE 模块整体缩放强度的平均水平

  ```math
  \mathrm{mean}^{(m)}=\frac{1}{C}\sum_{j=1}^{C}s^{(m)}_j
  ```

- `std`：不同通道之间的缩放差异有多大

- `min` / `max`：最弱通道和最强通道被压制或增强到什么程度

如果某个 SE 模块的：

- `std` 很小，且 `min / max` 都接近同一水平, 说明它对各通道几乎一视同仁，通道选择作用较弱
- `std` 较大，且 `min / max` 差距明显，说明它确实在做更强的通道重标定，不同通道被赋予了明显不同的权重

### 9.6 UMAP

UMAP（Uniform Manifold Approximation and Projection）用于把模型最后的高维 embedding 非线性嵌入到二维平面，方便观察不同类别在特征空间中的分布结构

它和 PCA 的关注点不同

PCA 是线性降维，它寻找一组正交方向，使投影后的整体方差尽可能大

PCA 更像是在问哪些线性方向能保留最多整体方差信息

如果 embedding 中的类别结构主要能被少数线性方向解释，PCA 会很有效

但如果类别边界是非线性的，PCA 可能会把局部类别结构压扁

UMAP 是非线性流形降维，它不是直接寻找最大方差方向，而是先在高维空间中构建近邻图，再在二维空间中寻找一个布局，使高维近邻关系在低维空间中尽量保持

它更像是在问高维空间里谁和谁是邻居，二维图里能不能也尽量保持这种邻近关系

设高维 embedding 的近邻图为：

```math
\mathcal{G}_{\mathrm{high}}
=
(V,E,w_{ij})
```

其中 $V$ 是样本点，$E$ 是近邻边，$w_{ij}$ 表示样本 $i$ 和样本 $j$ 在高维空间中的邻近强度

UMAP 要寻找二维表示 $y_i \in \mathbb{R}^2$，使低维近邻图尽量接近高维近邻图：

```math
Y^*
=
\arg\min_Y
D\left(
\mathcal{G}_{\mathrm{high}},
\mathcal{G}_{\mathrm{low}}(Y)
\right)
```

这里 $D(\cdot)$ 是概念化写法，用来表示高维近邻结构和低维近邻结构之间的差异

实际 UMAP 会通过优化近邻图结构相似性来得到二维布局

分析阶段会把模型切换到 eval 模式，然后分别遍历训练集和测试集

```python
model.eval()
```

对每个 batch，模型前向时打开 `return_feat=True`

```python
_, feat = model(x, return_feat=True)
```

设第 $n$ 个样本的 embedding 为：

```math
z_n \in \mathbb{R}^{D}
```

所有训练集和测试集样本拼接后得到：

```math
Z \in \mathbb{R}^{N_{all} \times D}
```

```python
reducer = umap.UMAP(
    n_neighbors=actual_neighbors,
    min_dist=min_dist,
    n_components=2,
    random_state=random_state,
)
emb_2d = reducer.fit_transform(feats)
```

- `n_components=2`：输出二维坐标
- `n_neighbors`：控制局部邻域大小
- `min_dist`：控制二维图中点云的紧凑程度
- `random_state=42`：固定随机种子，保证图像尽量可复现

把 train 和 test 放在同一个 UMAP 坐标系中，因为如果 train 和 test 分别单独做 UMAP，两个图的坐标不能直接比较，当前做法是先联合降维，再拆成左右两个子图，并共享坐标轴，可以直接看测试集是否和训练集对齐

常见现象解释：

1. **同类样本聚在一起**

   说明模型对该类学到了较稳定的 embedding 表征

2. **不同颜色明显分开**

   说明父层级类别在特征空间中可分性较好

3. **同色不同 marker 也能分开**

   说明同一父类下的子类也具有一定可分性

4. **训练集分得开，但测试集混在一起**

   可能存在测试集分布偏移、采集条件差异或预处理不一致

5. **训练集和测试集都混在一起**

   说明当前层级本身在 embedding 空间中没有被明显拉开，可能是类别差异弱、样本不足或模型表征不够

6. **某个类别形成多个小簇**

   可能说明该类别内部存在批次差异、小文件夹差异或采集条件差异，需要结合原始数据来源继续检查

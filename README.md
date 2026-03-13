# 拉曼光谱层级分类项目（细菌 + 耐药）

本文档按当前代码实现整理项目的完整技术细节，覆盖：

- 数据预处理
- 数据集构建与层级标签系统
- 模型结构（ResNeXt + Transformer/LSTM）
- 训练流程（层级训练、损失、调度、早停）
- 级联预测
- 测试评估
- 可解释性分析
- PCA+SVM 对照基线

更新时间：`2026-03-04`

---

## 1. 项目目标与任务定义

项目用于拉曼光谱分类，包含两类典型任务：

- 细菌 `species` 分类
- 耐药 `phenotype` 分类

核心特点：

- 统一训练入口：`train.py`
- 自动从目录结构构建层级标签（`level_1 ... level_N + leaf`）
- 支持按父类拆分训练子模型（`train_per_parent=True`）
- 推理时按层级级联，支持缺失子模型回退

---

## 2. 端到端技术路线

```text
原始 .arc_data
  -> (离线) baseline校正 + 截断 + 插值 + 坏波段剔除 + PCA异常值过滤
  -> dataset/<分类名>/dataset_train 或 dataset/<分类名>/dataset_test
  -> (在线) RAW域增强 -> 标准化 -> SNV后增强 -> 多通道构建
  -> ResNeXt1D 主干 -> 序列编码器(Transformer/LSTM/None) -> 池化(attn/stat) -> 分类头(cosine/linear)
  -> 训练：Focal + Align + SupCon (+DRW/EMA动态权重)
  -> 输出：多层模型、parent子模型、层级元数据、split清单
  -> 推理：层级级联 + parent mask
  -> 评估/解释：指标、混淆矩阵、IG/GradCAM/Embedding/波段热图
```

---

## 3. 仓库结构与模块职责

```text
拉曼光谱分类/
├─ train.py                          # 统一训练入口（含层级训练逻辑）
├─ raman/
│  ├─ config.py                      # 训练配置（唯一配置源）
│  ├─ config_io.py                   # config.yaml读写、实验重载
│  ├─ dataset.py                     # 层级数据集、标签映射、通道构建
│  ├─ preprocess.py                  # 在线预处理与两阶段增强
│  ├─ model.py                       # ResNeXt1D + (Transformer/LSTM) + 分类头
│  └─ train_utils.py                 # split/评估/损失/采样器工具
├─ predict/
│  ├─ predict_core.py                # 级联推理核心
│  ├─ predict_folder.py              # 批量目录预测
│  └─ predict_single.py              # 单目录预测（文件级输出）
├─ evalute/
│  └─ evalute_test.py                # 测试集评估（注意目录名是 evalute）
├─ analysis/
│  ├─ analysis_utils.py              # IG/GradCAM/embedding等分析底层实现
│  ├─ analysis_core.py               # 单模型/聚合分析主流程
│  ├─ analyze_single.py              # 单模型分析入口
│  └─ analyze_aggregate.py           # 按parent聚合分析入口
├─ dataset_process/                   # 统一离线预处理与数据集整理入口
├─ PCA+SVM/
│  └─ pca_svm_from_split.py          # 传统基线（按训练split复现）
└─ dataset/
   ├─ 细菌/
   ├─ 耐药菌/
   └─ 厌氧菌/
```

---

## 4. 数据格式与层级标签系统

### 4.1 输入文件格式

- 文件后缀：`.arc_data`
- 内容：两列数值
- 第1列：波数（wavenumber）
- 第2列：强度（intensity）

### 4.2 目录到标签映射（`raman/dataset.py`）

`RamanDataset` 扫描数据根目录后自动生成：

- `level_names`: `level_1 ... level_N`
- `head_names`: `level_1 ... level_N + leaf`
- `label_maps_by_level`: 每层 `name -> id`
- `inv_label_maps_by_level`: 每层 `id -> name`
- `parent_to_children`: `parent_id -> child_ids`
- `level_labels`: 每个样本的多层标签矩阵（无效为 `-1`）
- `hier_names`: 每个样本的层级路径字典

### 4.3 split 机制（防泄漏）

训练/测试划分调用：

- `split_by_lowest_level_ratio(dataset, lowest_level=split_level, train_ratio, seed)`

行为：

- 按 `split_level` 分组后切分（默认 `leaf`）
- 每个组独立打乱
- 单样本组默认放入训练集，避免测试集空组
- 切分结果保存为 `train_files.json` / `test_files.json`

这保证后续评估、基线、分析都能复用同一 split。

---

## 5. 离线数据预处理（`dataset_process` + `dataset/<分类名>`）

不同数据集共用一套流程，差异参数（例如 `BAD_BANDS`）集中在 `dataset_process/profiles.py`。

### 5.1 阶段1：原始目录重组（`classify_dataset.py`）

输入支持：

- 目录 `dataset_init/`
- 压缩包 `dataset_init.npz`（通过 `packed_dataset.py`）

处理：

- 从叶子目录名提取前缀作为类别（细菌：字母前缀；耐药：支持 `+/-`）
- 文件重命名为 `叶子名_原文件名`
- 输出到 `dataset_raw/`

### 5.2 阶段2：训练集预处理（`preprocess_dataset.py`）

每条光谱执行：

1. 读取 `.arc_data`
2. AsLS 基线校正
3. 波段截断（`CUT_MIN`~`CUT_MAX`）
4. 插值到 `TARGET_POINTS`
5. 剔除坏波段 `BAD_BANDS`
6. （可选）PCA 重构误差异常值过滤（按 `PCA_OUTLIER_RATIO` 去除尾部）

类别样本数 `< MIN_SAMPLES_PER_CLASS` 会跳过。

输出：

- 清洗后光谱到 `dataset/<分类名>/dataset_train`
- 类别均值谱图到 `dataset/<分类名>/dataset_train_fig/`
- PCA剔除日志 `log.txt`

### 5.3 阶段3：测试集预处理（`preprocess_testdata.py`）

流程与训练集一致，但不做 PCA 异常值剔除，输出到 `dataset/<分类名>/dataset_test`，并保存类别均值谱图到 `dataset/<分类名>/dataset_test_fig/`。

### 5.4 其他脚本

- `count_dataset.py`：递归统计各层目录样本数
- `pack_dataset_init.py` + `packed_dataset.py`：将 `dataset_init` 打包/读取为 `npz`，便于迁移和存储

---

## 6. 在线预处理与增强（训练/推理时）

实现文件：

- `raman/preprocess.py`
- `raman/dataset.py`
- `raman/preprocess.py::InputPreprocessor`（预测/评估一致预处理）

### 6.1 训练样本主流程（`RamanDataset.__getitem__`）

1. 读取强度列
2. 训练时 RAW 域增强（`augment_raw_spectrum`）
3. 标准化（`snv/l2/minmax`）
4. 训练时 SNV 后增强（`augment_norm_spectrum`）
5. 构建输入通道（`snv_posneg_split` + `smooth` + `d1`）

推理/评估时不会启用增强，只保留与训练一致的裁剪、插值、标准化和通道构建逻辑。

### 6.2 RAW 域增强（pre-augment）

函数：`augment_raw_spectrum`

这一阶段的目标不是“凭空造新谱”，而是模拟采集条件和仪器响应变化。实现上分成几组候选操作：

- `aug_piecewise_gain`：
  将整条谱随机切成若干区段，每段乘一个接近 `1.0` 的随机比例，模拟不同 Raman band 相对强度变化。
- `aug_noise_gaussian`：
  噪声强度按当前谱的稳健振幅缩放，不是固定绝对值，避免强谱和弱谱加同样大小噪声。
- `aug_noise_poisson`：
  噪声幅度与局部强度相关，近似模拟“信号越强，噪声也越大”的采集特征。
- `aug_axis_warp`：
  用“小线性漂移 + 低频正弦扰动 + 插值回原网格”的方式模拟波数轴轻微标定误差。
- `aug_weak_baseline`：
  用线性项和低频正弦项叠加出弱背景，模拟 baseline 去除不干净。
- `aug_strong_baseline`：
  通过少量控制点插值构造低频弯曲基线，模拟跨批次或跨仪器背景差异。

执行策略：

- `piecewise_gain`、`axis_warp` 独立按概率决定是否加入
- `gaussian / poisson` 二选一，避免两种噪声同时叠加
- `weak_baseline / strong_baseline` 二选一，避免同一次增强里叠两套背景
- 采样出的操作会随机打乱顺序，再截断到 `max_pre_augs`
- 全部实现都按有效波段工作，坏段或 `NaN` 不会被拿去造假信息

### 6.3 SNV 后增强（post-augment）

函数：`augment_norm_spectrum`

这一阶段不再模拟整体强度，而是针对“谱形”和“局部峰形”做轻扰动：

- `aug_shift`：
  以整数点为单位整体微移，模拟重复测量时峰位轻微偏差。
- `aug_broadening`：
  对谱做小核高斯卷积，让局部峰轻微展宽，但不会把整条谱抹平。
- `aug_mask_attenuate`：
  随机选一段区间，用带平滑边缘的衰减窗压低局部响应，模拟局部污染、坏点或局部失真。

执行策略：

- 每种操作独立按概率采样
- 采样后随机打乱，再截断到 `max_post_augs`
- 这类增强发生在标准化之后，主要改变形状与局部模式，而不是整体能量尺度
- 因此它更像“提高重复测量鲁棒性”，而不是模拟跨仪器强度漂移

### 6.4 通道构建

默认支持：

- `snv_posneg_split=True`：`pos=max(x,0)` + `neg=max(-x,0)`
- `smooth_use=True`：SG 平滑通道
- `d1_use=True/False`：一阶导通道（按 `delta` 缩放并归一）

`in_channels` 由配置属性自动计算并与实际构建通道数强校验。

---

## 7. 模型架构细节（`raman/model.py`）

模型类：`ResNeXt1D_Transformer`

### 7.1 主干网络（ResNeXt1D + SE）

- Stem：
  - 单尺度 `Conv1d + BN + ReLU + AvgPool`
  - 或多尺度分支 `kernel_sizes=(3,7,15)` 后拼接，再 `AvgPool`
- 主干层：
  - `layer1(64x2)`、`layer2(128x2)`、`layer3(256x2)`、`layer4(384x2)`
  - 层间下采样使用 `AvgPool1d`
- 基本块 `ResNeXtBlock1D`：
  - `1x1` 降维
  - `3x3 group conv`
  - `1x1` 升维
  - `SEBlock1D`
  - shortcut 残差

### 7.2 序列编码器角色

- ResNeXt：局部峰形/局部组合模式提取（局部模式编码）
- Transformer/LSTM：长程依赖与全局上下文建模（跨峰关系编码）

当前可选：

- `backbone_type="cnn"`：使用 ResNeXt1D 主干
- `backbone_type="identity"`：跳过 CNN，直接平均下采样后做通道投影
- `encoder_type="transformer"`：位置编码 + TransformerEncoder
- `encoder_type="lstm"`：单/双向 LSTM
- `encoder_type="none"`：跳过序列编码，仅用前端特征

常用消融组合：

- 仅 CNN：`backbone_type="cnn"`，`encoder_type="none"`
- 仅 LSTM：`backbone_type="identity"`，`encoder_type="lstm"`
- 仅 Transformer：`backbone_type="identity"`，`encoder_type="transformer"`
- CNN + LSTM：`backbone_type="cnn"`，`encoder_type="lstm"`
- CNN + Transformer：`backbone_type="cnn"`，`encoder_type="transformer"`

### 7.3 池化与分类头

池化：

- `pooling_type="attn"`：可学习注意力加权
- `pooling_type="stat"`：`mean + std` 统计池化（输出维度翻倍）

分类头：

- 线性头：`Linear`
- 余弦头：`CosineClassifier`
  - 对特征和权重做 L2 归一化
  - 输出 `scale * cos(theta)`
  - 对幅值漂移更稳，方向判别更强

---

## 8. 损失函数与优化策略

实现文件：`raman/train_utils.py` + `train.py`

### 8.1 主损失：FocalLoss

`FocalLoss(gamma, class_weight, ignore_index=-1)`  
用于抑制易分类样本、增强难样本梯度。

### 8.2 可选难度再加权（`use_severity_weight`）

对每个样本的主损失再乘样本权重：

- top2/top3 命中真类可减轻惩罚
- 高置信错分（top1_conf>0.8 且错误）加重惩罚

### 8.3 Align Loss（层级中心收紧）

`hierarchical_center_loss(feat, hier_labels, level_weights)`  
按目标层级计算类内中心半径损失，增强类内紧凑性。

### 8.4 SupCon Loss（监督对比）

`SupConLoss(temperature=tau)`  
在 embedding 空间拉近同类、拉远异类。  
支持通过 `supcon_level` 指定对比标签层级，不必等于当前训练层。

### 8.5 DRW/EMA 动态类别权重（`use_drw`）

训练中维护每类 CE 的 EMA：

- `ema_class_ce`
- 根据难度比例生成 `dynamic_weights`
- 替换 Focal 的 `criterion.weight`

### 8.6 损失权重调度

- `align_w`、`supcon_w` 在 `[start, end]` 线性 warm-up
- 到 `decay_start_ratio * epochs` 后线性衰减，最小保持 `0.2`

`decay_start_ratio` 规则：

- 直接在 `raman/config.py` 中显式设置具体值
- 训练时按该值读取，并限制在 `[0.0, 1.0]`

### 8.7 优化器与学习率

- Adam + weight decay
- 参数分组学习率：
  - stem：`0.6 * lr`
  - backbone：`1.0 * lr`
  - head：`1.2 * lr`
- 调度器：`CosineAnnealingLR`

### 8.8 早停指标

`score = w_f1 * macro_f1 + w_acc * acc`  
超过历史最佳则保存模型，超过 `patience` 触发早停。

---

## 9. 训练流程（`train.py`）

### 9.1 顶部手动覆盖项（高频实验开关）

包括：

- `TRAIN_ONLY_LEVEL`
- `TRAIN_ONLY_PARENT_NAME` / `TRAIN_ONLY_PARENT`
- `OVERRIDE_DECAY_START_RATIO`
- `OVERRIDE_ALIGN_LOSS_WEIGHT`
- `OVERRIDE_SUPCON_TAU`
- `OVERRIDE_SUPCON_LOSS_WEIGHT`
- `SUPCON_LEVEL_OVERRIDE`
- `OVERRIDE_TIMESTAMP`
- `OVERRIDE_OUTPUT_DIR`

注意：当前代码中 `TRAIN_ONLY_LEVEL="level_1"`、`SUPCON_LEVEL_OVERRIDE="level_1"`。  
如果希望按 `config.train_level` 正常全流程训练，需将二者设为 `None`。

### 9.2 训练组织模式

- `train_per_parent=False`：每个层级一个全局模型
- `train_per_parent=True`：
  - 顶层（无父类）训练全局模型
  - 下层按父类训练子模型
  - 父类仅一个子类时跳过建模（推理直接确定）

### 9.3 训练过程关键点

- split 优先复用实验目录已有 `train_files.json/test_files.json`
- 支持 `train_filter_level/train_filter_value` 过滤训练子集
- 每个模型单独保存 best 权重
- 评估函数按模型类型选择：
  - 全局模型：`evaluate_file_level`（可父类mask）
  - 局部子模型：`evaluate_file_level_local`（全局id到局部id映射）

### 9.4 训练产物

每次实验目录通常包含：

- `config.yaml`：完整配置快照
- `logs/log.txt`：训练日志
- `class_names.json`：全层级类别名
- `hierarchy_meta.json`：层级结构和模型映射
- `level_x_model.pt` / `level_x_parent_k_model.pt`
- `train_files.json` / `test_files.json`

`hierarchy_meta.json` 关键字段：

- `head_names`
- `class_names_by_level`
- `parent_to_children`
- `parent_level_name`
- `train_level`
- `level_models`
- `parent_models`

---

## 10. 预测流程（`predict/predict_core.py`）

核心接口：

- `load_predictor(exp_dir, device, predict_level=None)`
- `predict_one(path, predictor, top_k=3, parent_mask=None)`

### 10.1 加载模式

- `single`：无 `hierarchy_meta.json` 时，加载单模型
- `cascade`：读取层级元数据，逐层级联

### 10.2 级联逻辑

每一层：

1. 若存在对应 parent 子模型，优先使用子模型
2. 否则回退全局层模型 + 父类mask
3. 若子模型缺失/无效，回退到“当前路径最深可用层级”结果

### 10.3 parent mask

`parent_mask` 支持：

- 按名称约束：`{"level_1": ["feike"]}`
- 按索引约束：`{"level_1": [0]}`

用于把预测限制在业务先验允许的类别范围。

---

## 11. 测试集评估（`evalute/evalute_test.py`）

### 11.1 可配置项（脚本顶部）

- `EXP_DIR`
- `EVAL_LEVEL`
- `EVAL_ONLY_LEVEL`
- `EVAL_ONLY_PARENT`
- `INHERIT_MISSING_LEVELS`

### 11.2 评估行为

- 优先使用训练时保存 split
- fallback 才按 `split_level` 重新切分
- 支持级联到目标层级并结合 parent 子模型
- `INHERIT_MISSING_LEVELS=True` 时，缺失层级可向 `leaf` 继承用于展示/统计

### 11.3 输出文件

在 `EXP_DIR/{EVAL_LEVEL}_test_result/`：

- `test_eval_results.csv`：样本明细
- `classification_report.txt`：类别精确率/召回/F1
- `confusion_matrix_raw.csv`：原始混淆矩阵
- `confusion_matrix.png`：归一化混淆矩阵热图

---

## 12. 可解释性分析（`analysis/`）

### 12.1 入口脚本

- 单模型/按parent分析：`analysis/analyze_single.py`
- parent聚合分析：`analysis/analyze_aggregate.py`
- 流程编排：`analysis/analysis_core.py`
- 可复用分析函数：`analysis/analysis_utils.py`

### 12.2 单模型分析输出

目录：`EXP_DIR/{tag}_analysis/`

- `figures/channel_importance_IG.png`
- `figures/layer_importance.png`
- `figures/{tsne|umap}_hier.png`
- `figures/{tsne|umap}_hier_train_test.png`
- `figures/band_importance_heatmap.png`
- `figures/band_topK_per_class.csv`
- `logs/analysis_log.txt`

### 12.3 聚合分析输出

目录：`EXP_DIR/{analysis_level}_aggregate_analysis/`

- `figures/channel_importance_IG_aggregate.png`
- `figures/layer_importance_aggregate.png`
- `figures/band_importance_heatmap_aggregate.png`
- `figures/band_topK_per_class_aggregate.csv`
- `logs/analysis_log.txt`

### 12.4 分析方法实现

- 输入通道重要性：Integrated Gradients（支持 mean baseline）
- 层重要性：多层 Grad-CAM 风格 `|A*G|` 聚合
- SE统计：读取 `SEBlock1D.latest_scale` 统计摘要
- 嵌入可视化：UMAP/TSNE（自动适配 sklearn TSNE 参数）
- 波段热图：按类平均 IG 重要性 + 均值谱叠加 + bad bands 标注

---

## 13. PCA+SVM 对照基线（`PCA+SVM/pca_svm_from_split.py`）

流程：

1. 从 `EXP_DIR` 读取 `config.yaml` 和 split 清单
2. 按指定层级提取样本特征
3. `StandardScaler` 标准化
4. PCA 降维（`n_components` 支持方差比例或固定维数）
5. SVM 分类（默认 RBF）
6. 输出分类报告与混淆矩阵

特征方式：

- `USE_ALL_CHANNELS=False`：只用第1通道
- `USE_ALL_CHANNELS=True`：展平特征使用全部通道

输出：

- `metrics.txt`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `pca_scatter.png`

当前已统一为百分比格式输出，`accuracy/macro avg/weighted avg` 保留 `4` 位小数百分比。

---

## 14. 配置使用原则（`raman/config.py`）

README 不再逐项抄写 `config.py` 的全部默认值；配置源码本身是唯一准确来源。这里更强调“怎么改”和“优先改什么”。

### 14.1 最常改的几组配置

- 任务与数据：
  `dataset_root`、`train_level`、`train_per_parent`、`split_level`、`BAD_BANDS`
- 输入与预处理：
  `norm_method`、`snv_posneg_split`、`smooth_use`、`d1_use`
- 模型结构：
  `backbone_type`、`encoder_type`、`pooling_type`、`cosine_head`
- 编码器容量：
  `transformer_dim / nhead / ffn_dim / layers`，或 `lstm_hidden / layers / bidirectional`
- 训练控制：
  `learning_rate`、`batch_size`、`epochs`、`patience`、`seed`
- 增强强度：
  `p_*` 系列概率、`max_pre_augs`、`max_post_augs`

### 14.2 结构开关的常用组合

- `仅CNN`：`backbone_type="cnn"`，`encoder_type="none"`
- `仅LSTM`：`backbone_type="identity"`，`encoder_type="lstm"`
- `仅Transformer`：`backbone_type="identity"`，`encoder_type="transformer"`
- `CNN+LSTM`：`backbone_type="cnn"`，`encoder_type="lstm"`
- `CNN+Transformer`：`backbone_type="cnn"`，`encoder_type="transformer"`

其中 `identity_pool_kernel` 只在 `backbone_type="identity"` 时生效，用来控制“非 CNN 路线”送进序列编码器前的长度压缩量。

### 14.3 推荐的调参顺序

1. 先固定数据入口、训练层级和 split 策略。
2. 再固定主结构组合，例如先确定是 `CNN+Transformer` 还是 `仅LSTM`。
3. 结构定下来后，再调编码器容量和 dropout。
4. 最后再调增强强度、损失权重和 early stop 相关参数。

### 14.4 关于增强参数的理解方式

- `RAW` 域参数主要控制“采集条件变化”的模拟强度。
- `SNV后` 参数主要控制“谱形局部扰动”的强度。
- 如果出现训练早期塌成单类、测试集长时间不动，优先检查结构与学习率，不要只靠加大增强概率硬顶。
- 如果训练集很高、测试集明显掉，优先考虑减弱 `strong baseline` 和 `cut` 这一类强扰动。

---

## 15. 运行手册（推荐顺序）

### 15.1 环境依赖（按代码导入）

- Python 3.9+
- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `pyyaml`
- `tqdm`
- `umap-learn`

### 15.2 离线预处理

所有 `dataset_process` 命令都在项目根目录执行：

```bash
python -m dataset_process <command> <数据集名>
```

支持的数据集名：

- `细菌`
- `耐药菌`
- `厌氧菌`

可用指令：

- `python -m dataset_process list`
- `python -m dataset_process pack-init 细菌`
- `python -m dataset_process unpack-init 细菌`
- `python -m dataset_process classify 细菌`
- `python -m dataset_process preview-init 细菌`
- `python -m dataset_process preprocess-train 细菌`
- `python -m dataset_process preprocess-test 细菌`
- `python -m dataset_process count 细菌`
- `python -m dataset_process count 细菌 --subdir dataset_raw`

指令说明：

- `list`：列出当前支持的数据集 profile。
- `pack-init`：将 `dataset/<数据集名>/dataset_init/` 打包为 `dataset_init.npz`。
- `unpack-init`：将 `dataset_init.npz` 解包回 `dataset_init/`。
- `classify`：从 `dataset_init/` 或 `dataset_init.npz` 分类整理到 `dataset_raw/`。
- `preview-init`：对 `dataset_init` 中每个原始叶子文件夹单独做预处理并输出图到 `dataset_init_fig/`，不会执行 PCA 异常点剔除。
- `preprocess-train`：从 `dataset_raw/` 生成 `dataset_train/` 和 `dataset_train_fig/`。
- `preprocess-test`：从 `测试菌/` 生成 `dataset_test/` 和 `dataset_test_fig/`。
- `count`：统计默认目录或指定子目录中的 `.arc_data` 数量。

常用流程示例：

```bash
python -m dataset_process unpack-init 厌氧菌
python -m dataset_process preview-init 厌氧菌
python -m dataset_process classify 厌氧菌
python -m dataset_process preprocess-train 厌氧菌
python -m dataset_process preprocess-test 厌氧菌
python -m dataset_process count 厌氧菌
```

### 15.3 训练

```bash
python train.py
```

训练前建议先检查：

- `raman/config.py`（任务路径、train_level、增强参数）
- `train.py` 顶部覆盖项（尤其 `TRAIN_ONLY_LEVEL`）

### 15.4 预测

```bash
python predict/predict_folder.py
python predict/predict_single.py
```

### 15.5 测试评估

```bash
python evalute/evalute_test.py
```

### 15.6 可解释性分析

```bash
python analysis/analyze_single.py
python analysis/analyze_aggregate.py
```

### 15.7 PCA+SVM 基线

```bash
python "PCA+SVM/pca_svm_from_split.py"
```

---

## 16. 常见坑位与排查建议

- 训练层级不符合预期：先检查 `train.py` 顶部覆盖变量是否仍启用。
- 推理找不到模型：确认 `EXP_DIR` 下 `config.yaml`、`hierarchy_meta.json`、`*_model.pt` 是否完整。
- 评估结果与训练不一致：确认是否复用了同一 `train_files.json/test_files.json`。
- 通道数报错：检查 `snv_posneg_split/smooth_use/d1_use` 与 `in_channels`。
- bad bands 长度差异：离线预处理 `BAD_BANDS` 应与训练配置保持一致。
- `evalute` 目录名是历史拼写（不是 `evaluate`），脚本路径注意不要写错。

---

## 17. 概念补充（与当前实现对应）

### 17.1 SupCon vs Align

- SupCon：同类拉近 + 异类拉远，强调判别边界。
- Align：主要做类内收紧，强调表示一致性。
- 本项目可同时启用，并通过 warm-up 与后期衰减降低过约束风险。

### 17.2 统计池化 vs 注意力池化

- `stat`：稳定、抗过拟合、对小样本友好。
- `attn`：表达更强，能突出关键波段，但更依赖数据规模与正则。

### 17.3 余弦分类头 vs 普通线性头

- 线性头：方向 + 幅值共同影响决策。
- 余弦头：方向主导，幅值影响弱，跨批次强度漂移下通常更稳。

---

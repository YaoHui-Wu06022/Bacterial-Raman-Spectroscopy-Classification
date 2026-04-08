# 拉曼光谱层级分类项目

## 1. 项目目标与任务定义

项目用于拉曼光谱分类，可实现多层级分类

核心特点：

- 统一训练入口：`train.py`
- 自动从目录结构构建层级标签
- 支持按父类拆分训练子模型（`train_per_parent=True`）
- 推理时按层级级联，支持缺失子模型回退

## 2. 仓库结构与模块职责

```text
拉曼光谱分类/
├─ train.py                          # 统一训练入口（含层级训练逻辑）
├─ evaluate_test_set.py              # 测试集评估入口
├─ pca_svm_baseline.py               # PCA+SVM 基线入口
├─ analyze.py                        # 统一分析入口（single / aggregate）
├─ compare_test_train_means.py       # 外部测试菌与训练类均值谱对比
├─ pack_raman.py                     # 打包 raman 库，便于上传到 Colab
├─ raman/
│  ├─ config.py                      # 训练配置
│  ├─ config_io.py                   # config.yaml 读写、实验重载
│  ├─ model.py                       # 可切换 ResNet/ResNeXt + Transformer/LSTM + 分类头
│  ├─ prototype.py                   # 原型向量保存与融合推理
│  ├─ trainer.py                     # 训练主流程与单模型训练实现
│  ├─ data/
│  │  ├─ paths.py                    # 数据目录阶段解析
│  │  ├─ dataset.py                  # 层级数据集、标签映射、通道构建
│  │  └─ preprocess.py               # 在线预处理与增强
│  ├─ analysis/
│  │  ├─ core.py                     # 单模型/聚合分析主流程
│  │  └─ utils.py                    # IG、GradCAM、embedding 等分析底层实现
│  ├─ eval/
│  │  ├─ experiment.py               # 实验路径、配置和层级名解析
│  │  ├─ report.py                   # 分类报告与混淆矩阵输出
│  │  ├─ test_set_evaluator.py       # 测试集评估实现
│  │  └─ baseline.py                 # PCA+SVM 基线实现
│  └─ training/
│     ├─ split.py                    # 数据切分与训练范围解析
│     ├─ eval.py                     # 训练期评估工具
│     ├─ losses.py                   # 损失函数与权重工具
│     ├─ session.py                  # 训练会话、输出配置与复现控制
│     └─ sampler.py                  # 分层采样器
├─ dataset_process/
│  ├─ cli.py                         # 离线数据处理统一入口
│  ├─ profiles.py                    # 不同数据集的目录和坏波段配置
│  ├─ common.py                      # 离线预处理公共函数
│  └─ pipeline.py                    # 打包、重组、清洗、统计主流程
├─ predict/
│  ├─ predict_core.py                # 级联推理核心
│  ├─ predict_folder.py              # 批量目录预测
│  └─ predict_single.py              # 单目录预测
├─ colab/
│  └─ colab_unified.ipynb            # Colab 训练/评估/分析一体化 notebook
├─ notebooks/
│  └─ single_process_AsLS_cut_SNV.ipynb # 单条光谱处理流程可视化 notebook
└─ dataset/
   ├─ 细菌/
   ├─ 耐药菌/
   ├─ 厌氧菌/
```

## 3. 离线数据预处理

离线数据预处理统一走 `dataset_process`，当前常用命令只有：

- `pack-init`
- `unpack-init`
- `classify`
- `preview-init`
- `preprocess-train`
- `preprocess-test`
- `count`

### 3.1 参数修改位置

当前离线清洗参数不从 CLI 传入，统一在 `dataset_process/pipeline.py` 里修改：

- `DEFAULT_PIPELINE_CONFIG`

这里集中控制：

- 波段裁剪范围 `cut_min` / `cut_max`
- 统一参考波数轴点数 `target_points`
- AsLS 参数 `asls_lam` / `asls_p` / `asls_max_iter`
- 训练集最小样本数 `min_samples_per_class`
- 绘图归一化方式 `norm_method`
- PCA 异常值过滤相关参数

不同数据集的目录名在 `dataset_process/profiles.py` 里维护；坏波段现在全局固定为 `890~950 cm^-1`

补充说明：

- `target_points=896` 指的是完整参考波数轴长度
- 当前流程会在参考轴上同步删掉 `890~950 cm^-1`，因此最终落盘光谱长度不是 `896`，而是 `851`

### 3.2 输入目录与输出目录

以 `dataset/细菌` 为例，离线流程主要使用这些目录：

- `dataset_init/`：原始按测量文件夹组织的数据
- `dataset_init.npz`：`dataset_init/` 的打包版本
- `dataset_train_raw/`：按类别前缀重组后的中间结果
- `dataset_train/`：训练集离线清洗结果
- `dataset_test_raw/`：测试集原始输入目录
- `dataset_test/`：测试集离线清洗结果
- `dataset_train_fig/`：训练集均值谱图
- `dataset_test_fig/`：测试集均值谱图
- `dataset_init_fig/`：原始数据预览图
- `log.txt`：训练集 PCA 异常值剔除日志

### 3.3 阶段 1：打包与解包

先把 `dataset_init/` 打成一个压缩包：

```bash
python -m dataset_process pack-init 细菌
```

需要恢复成目录时：

```bash
python -m dataset_process unpack-init 细菌
```

### 3.4 阶段 2：原始目录重组

```bash
python -m dataset_process classify 细菌
```

作用：

- 扫描 `dataset_init/` 或 `dataset_init.npz`

- 读取叶子目录名

- 统一按 `letters_sign` 规则提取类别前缀
  
  例如：`ABC12 -> ABC`，`ESBL+03 -> ESBL+`
  
- 将样本复制或写出到 `dataset_train_raw/`

- 输出文件名统一改成 `叶子目录名_原文件名`

这一步的目的是先把原始采集目录整理成更稳定的类别目录结构，供后续统一清洗

### 3.5 阶段 3：原始数据预览

```bash
python -m dataset_process preview-init 细菌
```

作用：

- 直接基于 `dataset_init/` 或 `dataset_init.npz` 做预处理预览
- 执行基线校正、裁剪、坏波段剔除与统一参考轴插值
- 不做 PCA 异常值过滤
- 不落盘清洗后的光谱
- 只输出每个分组的均值谱图到 `dataset_init_fig/`

这一步适合先检查原始数据质量和类别分布是否合理

### 3.6 阶段 4：训练集离线清洗

```bash
python -m dataset_process preprocess-train 细菌
```

每条光谱执行：

1. 读取 `.arc_data`
2. AsLS 基线校正
3. 波段裁剪
4. 在裁剪后的原始波数轴上删除 `890~950 cm^-1` 坏段
5. 只对保留的统一目标波数轴做线性插值，不跨坏段补点
6. 对同一分组样本按 PCA 重构误差做异常值过滤

补充说明：

- 如果某个分组预处理后样本数少于 `min_samples_per_class`，该分组会跳过
- 被 PCA 剔除的样本会记录到 `log.txt`
- 当前默认 `cut_min=600`、`cut_max=1800`、`target_points=896`，最终写入 `dataset_train/` 的每条谱长度为 `851`

输出：

- 清洗后光谱写入 `dataset_train/`
- 每个分组的均值谱图写入 `dataset_train_fig/`

### 3.7 阶段 5：测试集离线清洗

```bash
python -m dataset_process preprocess-test 细菌
```

作用：

- 从 `dataset_test_raw/` 读取测试原始数据
- 执行基线校正、裁剪、坏波段剔除与统一参考轴插值
- 不做 PCA 异常值过滤
- 输出到 `dataset_test/`
- 同时为每个测试文件夹生成均值谱图到 `dataset_test_fig/`

### 3.8 数据统计

```bash
python -m dataset_process count 细菌
```

默认统计 `count_root` 指向的目录，一般是 `dataset_train/`

如果要统计其他子目录，可以用：

```bash
python -m dataset_process count 细菌 --subdir dataset_train_raw
```

输出是按目录层级展开的样本数统计，方便检查重组和清洗结果

## 4. 训练数据输入

### 训练数据处理流程

在数据离线处理后，`raman` 负责在线标准化、增强和模型输入通道构造

1. `RamanDataset` 从 `dataset_train/` 扫描目录树，自动构建：
   - `level_1 ... level_N`
   - `leaf`
   - 每层类别名与类别 id 映射
   - `parent_to_children`
2. 训练时 `__getitem__()` 读取单个 `.arc_data`，只取强度列
3. 强度列进入 `build_model_input()`，按当前配置构造成模型输入
4. `DataLoader` 将单条样本堆叠成 batch，送入模型

### 在线预处理与增强

`raman/data/preprocess.py` 当前的在线处理顺序是：

1. 读取离线清洗后的单条强度光谱
2. 如果是训练集并且 `augment=True`，先做 RAW 域增强
3. 按 `norm_method` 做标准化，默认是 `snv`
4. 如果是训练集，再做标准化后的形状增强
5. 构造模型输入通道

其中 RAW 域增强主要模拟采集条件变化，包括：

- 分段峰强比例扰动
- 高斯噪声或强度相关噪声
- 波数轴扰动
- 弱 / 强 baseline 扰动

标准化后的增强主要模拟重复测量和局部形状波动，包括：

- 峰位平移
- 峰展宽
- 局部衰减遮挡

### 模型输入

标准化后的单通道光谱不会直接送进模型，而是会按配置构造成多通道输入：

- 主通道是按 `norm_method` 标准化后的光谱，当前默认是 `snv`

- 如果开启 `smooth_use`，再增加一个 `smooth` 通道
  
  这个通道来自“RAW 增强后 -> SG 平滑 -> 标准化”，不再额外叠加标准化后的增强
  
- 如果开启 `raw_use`，再增加一个 `raw` 通道
  
  这个通道保留 RAW 增强后的未标准化输入，用来补充绝对强度和原始形状信息
  
- 如果开启 `d1_use`，再增加一个 `d1` 通道
  
  这个通道来自“RAW 增强后 -> 先平滑 -> 求一阶导 -> 标准化”，同样不再额外叠加标准化后的增强

因此单条样本最终形状是：

```text
[C, L]
```

其中：

- `C` 是输入通道数，由配置决定

- `L` 是离线统一后的光谱长度
  
  当前默认离线流程下，`L=851`

经过 `DataLoader` 后，模型实际接收的是：

```text
[B, C, L]
```

### 训练集、验证集、测试集

- 训练入口使用的基础数据目录是 `dataset_train/`
- 训练集和验证集都是从 `dataset_train/` 内部分割得到
- 如果实验目录下已有 `train_files.json` 和 `test_files.json`，会优先复用原切分
- 如果没有，就按 `split_level` 重新分组切分

当前训练代码中：

- `train_dataset = RamanDataset(..., augment=True)`，用于训练
- `test_dataset = RamanDataset(..., augment=False)`，用于训练过程中的验证

`test_dataset` 其实是“验证集视角”，不是外部测试集

独立测试集位于dataset下，另外使用`predict`预测

## 5. 模型

### 5.1 总体结构

当前模型定义在 `raman/model.py`，整体结构可以写成：

```text
输入 [B, C, L]
→ backbone（cnn / identity）
→ encoder（transformer / lstm / none）
→ pooling（attn / stat）
→ classifier（cosine / linear）
```

其中：

- `B`：batch size
- `C`：输入通道数，由 `smooth_use`、`raw_use`、`d1_use` 决定；当前默认是 `3`
- `L`：离线统一后的光谱长度

这套结构的设计目标不是做一个完全通用的 1D 分类器，而是围绕拉曼光谱的特点，把局部峰形建模、跨峰关系建模和最终判别头拆开，便于做消融实验。

### 5.2 Backbone

前端特征提取器由 `backbone_type` 控制：

- `cnn`：使用 `RamanClassifier1D` 内部的 1D CNN 主干
- `identity`：跳过 CNN，仅做平均下采样和 `1x1` 通道投影

默认配置使用 `cnn`，即：

- `cnn_block_type="resnext"`，也可切换到 `resnet`
- 多尺度 stem：`stem_multiscale=True`
- `stem_kernel_sizes=(3, 7, 15)`
- `backbone_activation="leaky_relu"`
- `cardinality=4`
- `base_width=4`
- `reduction=8`
- `se_use=True`

其中 CNN 路径的结构是：

1. 输入 stem  
   - 单尺度时：`Conv1d + BN + Activation + AvgPool1d`
   - 多尺度时：并联多个不同卷积核的 stem 分支，再在通道维拼接后统一池化
2. 四个 residual bottleneck stage  
   - 每个 stage 由两个 `ResidualBottleneck1D` 组成
   - stage 之间通过 `AvgPool1d` 做时序下采样
3. `1x1 Conv` 投影到统一的 `transformer_dim`

当 `cnn_block_type="resnext"` 时，中间 `3x3` 卷积使用 group convolution；当 `cnn_block_type="resnet"` 时则退化为普通 bottleneck 结构。

`ResidualBottleneck1D` 自身采用：

- `1x1` 降维
- `3x3` group convolution
- `1x1` 升维
- `SEBlock1D`
- residual shortcut

这里的多尺度 stem 对拉曼光谱比较重要，因为不同宽度的卷积核分别更擅长：

- 捕捉尖锐峰和窄局部结构
- 捕捉中尺度的峰群关系
- 捕捉更宽的缓变背景与峰包络

### 5.3 Encoder

前端 backbone 输出的是 `[B, C, L]` 形式的时序特征，之后会转成 `[B, L, C]`，再交给序列编码器

`encoder_type` 支持三种模式：

- `transformer`
- `lstm`
- `none`

默认配置使用 `transformer`，参数是：

- `transformer_dim = 192`
- `transformer_nhead = 6`
- `transformer_ffn_dim = 384`
- `transformer_layers = 1`
- `transformer_dropout = 0.2`

这一层的作用不是重新做全部局部峰提取，而是在 backbone 已经提炼出局部响应之后，继续建模不同波段之间的上下文关系

对拉曼光谱来说，可以理解为：

- backbone 更像“先找出哪些局部峰形有响应”
- encoder 更像“让峰 A 感知峰 B 是否同时出现，以及这些峰之间的组合关系”

### 5.4 Pooling

时序编码完成后，还需要把整条光谱压缩成一个固定长度的 embedding

`pooling_type` 支持：

- `attn`：注意力池化
- `stat`：统计池化（`mean + std`）

默认配置使用 `stat`，即：

```text
feat = concat(mean(out, dim=1), std(out, dim=1))
```

这样做的好处是：

- `mean` 保留整体平均激活水平
- `std` 保留不同波段响应的离散程度
- 对拉曼这种“峰值分布 + 局部变化幅度”都重要的信号，通常比单纯平均池化更稳

### 5.5 Classifier

最终分类头由 `cosine_head` 控制：

- `True`：`CosineClassifier`
- `False`：普通 `Linear`

当前默认配置使用线性分类头：

- `cosine_head=False`

主要是避免和后续预测时采用的`prototype`融合产生重叠

若开启余弦头，其核心思想是：

- 先对 embedding 和分类权重都做 L2 归一化
- 再计算余弦相似度
- 最后乘以一个可调的缩放系数 `scale`

这种做法的优点是：

- 更强调角度而不是绝对范数
- 和 SupCon 这类基于 embedding 几何关系的损失更一致
- 对长尾类别和类间边界较近的任务通常更稳

### 5.6 当前默认组合

当前默认配置下，实际训练的模型组合是：

```text
多通道输入
→ 多尺度 ResNeXt1D backbone
→ Transformer encoder
→ Statistical Pooling
→ Linear Classifier
```

这套默认组合背后的思路是：

- 用多通道输入显式提供不同光谱视角
- 用 CNN 先提峰形和局部结构
- 用 Transformer 建模跨波段关系
- 用统计池化保留整体均值和波动信息
- 用线性头完成当前层级分类

此外，训练结束后还可以基于训练集 embedding 额外保存一份 `*_prototypes.pt`，并在预测/评估时按配置切换：

- `classifier`：仅使用分类头输出
- `fusion`：分类头与 prototype 相似度融合
- `prototype`：仅使用 prototype 相似度

它本质上是一套“先局部建模，再全局关联，最后在 embedding 空间判别”的光谱分类模型

## 6. 训练

### 6.1 训练入口与实验目录

训练统一从根目录的 `train.py` 进入。

训练入口需要显式设置：

- `DATASET_NAME`
- `CURRENT_TRAIN_LEVEL`
- 是否只训练某个父类分支
- 若干手动覆盖项（如时间戳、输出目录、SupCon 权重等）

运行时会自动生成实验目录：

```text
output/<数据集名>/<时间戳>/
```

例如：

```text
output/细菌/20260330_153000/
```

实验目录内会保存：

- `config.yaml`
- `logs/config_<timestamp>.txt`
- `logs/run_<timestamp>.log`
- `logs/<model_tag>_<timestamp>.log`
- `train_files.json`
- `test_files.json`
- `class_names.json`
- `hierarchy_meta.json`
- 各层级或各父类对应的模型权重
- 各层级或各父类对应的 `*_prototypes.pt`

其中 `hierarchy_meta.json` 很重要，它记录了：

- 层级顺序
- 每层类别名
- `parent_to_children`
- 本次训练得到的全局模型、parent 子模型和 prototype 文件名

后续预测、评估和分析都会复用这些元数据

### 6.2 层级训练逻辑

训练入口里设置的是 `CURRENT_TRAIN_LEVEL`，但这里的含义不是“数据集中只有这一层”，而是：

- 数据集层级始终由 `dataset_train/` 目录树自动扫描得到
- `CURRENT_TRAIN_LEVEL` 只表示“这次训练实际要训练的那一层”

例如：

- 若 `CURRENT_TRAIN_LEVEL = "level_1"`，就只训练顶层模型
- 若 `CURRENT_TRAIN_LEVEL = "level_3"`，这一次只训练 `level_3`
- 若 `CURRENT_TRAIN_LEVEL = "leaf"`，就表示训练当前数据集实际存在的最细层级

也就是说，当前训练入口的行为等价于过去“只训练某一层”的模式，而不是自动从顶层一路训练到目标层。

当 `train_per_parent=True` 时，训练行为是：

- 顶层没有父层，因此训练全局模型
- 对更细层级，如果某层有父层，就按父类拆成多个子模型分别训练
- 如果某个父类下只有一个子类，则不训练该 parent 子模型，只在元数据中记录这条确定关系

如果当前实验目录缺少上一级模型或单子类记录，训练开始时会打印提示，提醒先训练哪一级。

这样做的好处是：

- 可以降低细粒度层级的类别混淆
- 让下层模型只在当前父类的候选子类范围内学习
- 在预测和评估时自然形成层级级联

### 6.3 训练/验证切分

训练代码扫描的是 `dataset_train/`，然后在内部再做 train/val 切分

切分逻辑位于 `raman/training/split.py`：

1. 优先检查实验目录里是否已有 `train_files.json` 和 `test_files.json`
2. 如果已有，则直接复用旧切分
3. 如果没有，则按 `split_level` 分组后再做比例切分

默认配置：

- `split_level = "leaf"`
- `train_split = 0.8`
- `seed = 42`

按 `leaf` 分组再切分的目的是尽量避免同一来源分组的样本同时落入训练集和验证集，从而减少信息泄漏

### 6.4 优化器、学习率和早停

当前训练器使用：

- `Adam`
- `weight_decay = 5e-4`
- `CosineAnnealingLR`

并按模块做了分组学习率：

- 输入 stem：`0.6 × learning_rate`
- backbone 其他部分：`1.0 × learning_rate`
- 分类头：`1.2 × learning_rate`

默认主学习率为：

```text
learning_rate = 4e-4
```

学习率调度参数：

- `scheduler_Tmax = epochs`
- `scheduler_eta_min = 1e-5`

早停评分不是单纯看验证集准确率，而是：

$$
\text{score} = w_{f1} \cdot \text{MacroF1} + w_{acc} \cdot \text{Accuracy}
$$

默认权重：

- `early_stop_w_f1 = 0.6`
- `early_stop_w_acc = 0.4`

也就是说，模型选择时会稍微更偏向宏平均 F1，而不是只偏向头部类别的整体准确率

### 6.5 训练增强

训练阶段的数据增强仍然来自 `raman/data/preprocess.py`，但在训练语境下可以更明确地分成两份数据流：

- `train_dataset = RamanDataset(..., augment=True)`
- `test_dataset = RamanDataset(..., augment=False)`

也就是说：

- 训练集看到的是“离线清洗后 + 在线增强”的输入
- 验证集看到的是“离线清洗后 + 仅标准化”的输入

当前在线增强分成两段：

1. RAW 域增强  
   - 分段峰强比例扰动
   - 高斯噪声 / 强度相关噪声
   - 波数轴扰动
   - baseline 扰动
2. 标准化后增强  
   - 峰位平移
   - 峰展宽
   - 局部衰减遮挡

其中 `smooth` 通道当前的构造方式是：

1. 基于 RAW 域增强后的光谱先做 SG 平滑
2. 再按 `norm_method` 做标准化
3. 不再额外叠加标准化后的增强

这样做的目的是让 `smooth` 通道更稳定地表达“平滑后的整体峰形”，避免它和主通道共享过多后增强造成语义混乱

其中 `d1` 通道当前的构造方式是：

1. 基于 RAW 域增强后的光谱先做 SG 平滑
2. 再求一阶导
3. 再按 `norm_method` 做标准化
4. 不再额外叠加标准化后的增强

另外当前默认还会保留一个 `raw` 通道，它直接使用 RAW 增强后的未标准化输入，用来补充绝对强度与原始谱形信息

### 6.6 当前训练总损失

训练时总损失由三部分组成：

$$
L_{\text{total}} = L_{\text{primary}} + \lambda_{\text{align}} L_{\text{align}} + \lambda_{\text{supcon}} L_{\text{supcon}}
$$

其中：

- `L_primary`：主分类损失，当前使用 `FocalLoss`
- `L_align`：层级中心损失，对应 `hierarchical_center_loss`
- `L_supcon`：监督式对比损失

`align` 和 `supcon` 两个辅助损失不是一开始就全权启用，而是：

- 在 `align_start ~ align_end`、`supcon_start ~ supcon_end` 区间线性拉升
- 在训练后期再按 `decay_start_ratio` 开始共同衰减

这样做的目的有两个：

- 前期先让模型把基本分类边界学稳
- 中期再逐步加强 embedding 结构约束

#### Focal Loss

在光谱层级分类任务中，不同样本难度差异较大，容易样本会主导梯度，导致模型忽略难样本

Focal Loss 在 CrossEntropy 的基础上增加一个可调节因子，抑制易样本梯度，放大难样本梯度，从而聚焦训练难样本

`CrossEntropy Loss`：

$$
CE(p_t) = - \log(p_t)
$$

`Focal Loss`：

$$
FL(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- $p_t$：模型对样本的预测概率 
- $\gamma$：控制对易样本的抑制程度
- $\alpha_t$：类别权重（可静态或动态，例如结合 DRW / EMA）  

```python
criterion = FocalLoss(
    gamma=config.gamma,               # 0.8
    weight=dynamic_weights,           # DRW / EMA 输出的类别动态权重
    ignore_index=-1,
    label_smoothing=config.label_smoothing
)
```

Focal Loss 关注样本难度 → 难样本梯度被放大，易样本梯度被抑制 

DRW / EMA 关注类别难度 → 类别层面的权重调整

在本项目中的使用方式：

- 主分类损失默认就是 `FocalLoss`
- `ignore_index=-1`，因此缺失标签不会参与当前层的主损失
- `label_smoothing` 默认关闭
- `gamma=0.8`，抑制强度相对温和，不是特别激进的 Focal 版本

除此之外，当前训练器还叠加了一层 `severity weight`：

- 若真实类别排到 top-2，样本权重下调到 `0.8`
- 若真实类别排到 top-3，样本权重下调到 `0.9`
- 若 top-1 高置信度预测错误且 `conf > 0.8`，样本权重提升到 `2.0`

这相当于在 Focal Loss 之外，再按“错得有多离谱”做一次样本级重加权。

#### class_weights

在进入 Focal Loss 之前，训练器会先根据当前训练层的标签分布构造一份基础类别权重 `class_weights`

当前实现不是直接使用简单的反频率，而是做了对数平滑：

1. 统计当前训练层每个类别的样本数
2. 对计数做下界保护，避免出现 0
3. 按下式计算基础权重：

   $$
   \text{weight}_g = \frac{1}{\log(\text{count}_g + 1.5)}
   $$

4. 再把所有类别权重归一化到平均值为 1：

   $$
   \text{weight}_g \leftarrow \frac{\text{weight}_g}{\frac{1}{C} \sum_{i=1}^{C} \text{weight}_i}
   $$

这么做的目的，是在照顾少数类的同时避免极端长尾下权重过大，导致训练不稳定

在本项目中的使用方式：

- `class_weights` 只根据当前训练层的训练样本计算
- 若当前训练的是父类内子模型，会先把全局标签映射到该子模型的局部标签空间，再统计类别分布
- 这份权重会先作为主分类损失的基础权重，再被后面的 `DRW / EMA` 进一步动态调整

可以把它理解成“静态的类别不平衡校正”

#### DRW / EMA class weight

在训练光谱层级分类模型时，类别分布通常不均衡，少数类样本容易被模型忽略

为缓解这一问题，采用动态类权重（Dynamic Re-weighting, DRW）结合指数移动平均（EMA）来动态调整每个类别的损失权重

1. 对每个类别在训练过程中计算当前 batch 的 CrossEntropy 平均损失

2. 用 EMA 平滑历史损失，公式为：

   $$
   \text{EMA}_g(t) = \alpha \cdot \text{EMA}_g(t-1) + (1-\alpha) \cdot \text{CE}_g^{\text{batch}}
   $$

   $\alpha$ 控制平滑程度（训练中为 0.9）

3. 根据 EMA 相对差异调整类别权重：

   $$
   \text{raw\_diff}_g = \frac{\text{EMA}_g(t)}{\frac{1}{C} \sum_{i=1}^{C} \text{EMA}_i(t)}
   $$

   $$
   \text{weight}_g = \text{class\_weight}_g \cdot \Big( 1 + \lambda (\text{raw\_diff}_g - 1) \Big)
   $$

   损失大的类别权重提升，损失小的类别权重降低

4. 归一化权重，保证平均为 1，以避免总梯度过大

   $$
   \text{weight}_g \leftarrow \frac{\text{weight}_g}{\frac{1}{C} \sum_{i=1}^{C} \text{weight}_i}
   $$

目的与作用：

- 平衡类间训练强度：少数类或难学类别在梯度中被放大，增强学习能力
- 平滑权重变化：EMA 让权重随训练逐步调整，避免单个 batch 异常值冲击训练
- 提高模型泛化能力：配合 Focal Loss 或 CrossEntropy，可显著提升长尾类别的识别准确率。

在本项目中的使用方式：

- 训练前先根据全训练集标签分布构造一份基础类权重 `class_weights`
- 从 `epoch >= 10` 开始启用 DRW/EMA
- 当前实现里：
  - `ema_momentum = 0.9`
  - `lambda_diff = 0.3`
- 每个 epoch 开始时，根据上一阶段累计的 `ema_class_ce` 更新 `criterion.weight`

也就是说，这里的 DRW / EMA 不是单纯按样本数静态加权，而是：

- 先考虑类别频次
- 再结合最近训练难度动态调整

这样能更好地区分“样本少但已经学会的类”和“样本少而且仍然学不好的类”。

#### severity weight

除了类别层面的重加权，当前训练器还会在样本层面再做一次“错误严重程度”加权

它的核心思路是：

- 如果模型虽然没把真实类排到第一，但已经排到 top-2 或 top-3，说明这个样本并不是完全错离谱
- 如果模型以很高置信度把样本分错，说明这是更危险的错误，应该放大梯度

当前实现里，训练器会对每个有效样本：

1. 计算当前 logits 的 softmax 概率
2. 取前 `k=min(3, num_classes)` 个预测类别
3. 统计真实类别在 top-k 中的排名
4. 按排名给样本损失乘一个额外权重

当前规则会按类别数自适应：

- 二分类：
  - 不额外做 `severity weight`
  - 权重保持 `1.0`
- 三分类：
  - 若真实类别排在 top-2，样本权重设为 `0.90`
  - 若高置信度错判且真实类别排在 top-2，权重提高到 `1.10`
  - 若高置信度错判且真实类别落到 rank-3，权重提高到 `1.45`
- 四类及以上：
  - 若真实类别排在 top-2，样本权重设为 `0.85`
  - 若真实类别排在 top-3，样本权重设为 `0.95`
  - 若高置信度错判且真实类别排在 top-2，权重提高到 `1.20`
  - 若高置信度错判且真实类别排在 rank-3 或更后，权重提高到 `1.80`

其中高置信度阈值也会按类别数调整：

- 三分类使用 `0.85`
- 四类及以上使用 `0.80`

因此这部分更准确地说，是一种“按预测错误严重程度调节主损失”的策略

- 降低“几乎分对”的样本对梯度的占用
- 提高“高置信度错判”样本的学习强度
- 和 `FocalLoss` 形成互补：`FocalLoss` 更关注概率难度，`severity weight` 更关注错误排序结构

#### SupCon Loss

核心思想：

- 在 embedding 空间约束样本的相对距离：
  - 同类样本靠近
  - 不同类样本远离
- 支持多模态类（一个类可以有多个簇），不要求类内单中心
- 主要用于表征学习，提高特征区分度

公式与实现：
1. 对 embedding 做 L2 正则化：

   $$
   z_i = \frac{feat_i}{||feat_i||_2}
   $$

2. 计算两两相似度矩阵：

   $$
   sim(i,j) = \frac{z_i \cdot z_j}{\tau}
   $$

   $\tau$：temperature，控制对比“硬度”

3. 构造 mask，确定正样本对（同类样本，`i != j`）

4. 数值稳定化，每行减去最大值

5. 计算 log-prob：

   $$
   \log p_{ij} = sim(i,j) - \log\sum_{k \neq i} e^{sim(i,k)}
   $$

6. 对每个 anchor 平均所有正样本：

   $$
   L_i = - \frac{1}{|P(i)|} \sum_{p \in P(i)} \log p_{ip}
   $$

7. Batch 平均：

   $$
   L = \frac{1}{B} \sum_i L_i
   $$

在本项目中的使用方式：

- SupCon 使用的是当前训练层级的标签，不再单独绑定一个配置层级
- 只有当前 batch 中存在至少两个有效同类样本时，SupCon 才会产生有效梯度
- 损失权重不是恒定的，而是通过 `supcon_start`、`supcon_end` 线性拉升
- 到训练后期，又会和 `align` 一起按 `decay_start_ratio` 逐步衰减

这样设计的考虑是：

- 前期先由主分类损失把类别边界学出来
- 中后期再用 SupCon 去整理 embedding 的几何结构
- 避免一开始就用对比损失把特征空间拉得过硬，影响分类头收敛

#### Center Loss

核心思想：
- 强制每个类别在 embedding 空间形成单一中心，最小化类内方差
- 对多层级标签可分别加权计算（`hierarchical_center_loss`）

公式：
$$
L = \frac{1}{N_c} \sum_{i=1}^{N_c} ||x_i - c_{y_i}||_2^2
$$

- $x_i$：样本 embedding
- $c_{y_i}$：该类别中心
- $N_c$：该类别样本数

特点：

- 类内紧凑性强，适合单模态类
- 对多模态类可能过强，导致信息损失
- 可与交叉熵联合使用，提高特征可分性

在本项目中的使用方式：

- 实现上对应的是 `hierarchical_center_loss`
- 当前训练某一层模型时，只对当前层级施加这一项约束
- 因此虽然函数支持“多层级分别加权”，但当前默认训练配置里实际是单层权重 `{level_name: 1.0}`
- 这项损失在训练日志里以 `AlignLoss` 记录

这么做的原因是：

- 当前层模型最直接的目标仍然是把本层 embedding 收紧
- 不在同一次训练里同时对所有层级都施加 center 约束，可以减少多层目标之间的梯度冲突

| 特性              | SupCon Loss                | Center Loss                  |
| ----------------- | -------------------------- | ---------------------------- |
| 类内约束          | 同类靠近（允许多簇）       | 强制单中心                   |
| 类间约束          | 间接推远不同类             | 不直接约束                   |
| 对 batch 大小敏感 | 高（需要足够正样本对）     | 低，单样本也能计算中心       |
| 使用场景          | 表征学习、对比学习、多模态 | 分类增强、特征紧凑化         |
| 复杂度            | 中等（矩阵相似度计算）     | 低（仅计算类中心和欧氏距离） |

| Loss                     | 约束目标           | 作用对象                       |
| ------------------------ | ------------------ | ------------------------------ |
| CrossEntropy / FocalLoss | 分类概率           | 样本层面，保证分类准确         |
| class_weights            | 静态类别重加权     | 类别层面，缓解基础长尾不平衡   |
| DRW / EMA                | 类别动态梯度权重   | 调整主分类损失梯度，针对少数类 |
| severity weight          | 样本错误严重程度   | 样本层面，突出高置信度错判     |
| SupCon Loss              | 类内靠近、类间分离 | embedding 层面，相对距离约束   |
| Center Loss              | 类内单中心         | embedding 层面，绝对距离约束   |

### 6.7 训练损失之间的分工

可以把当前训练里用到的几类损失理解成三层约束：

1. 主分类层`FocalLoss`：直接约束最终类别概率

2. embedding 几何层：
   - `SupCon Loss`
   - `Center Loss`
   
   约束特征空间内部结构
   
3. 类别重加权层  
   - `class_weights`
   - `DRW / EMA`
   - `severity weight`
   
   调整不同类别和不同难度样本的梯度贡献

它们并不是互相替代，而是：

- `FocalLoss` 负责“分对”
- `SupCon` 负责“拉开”
- `Center Loss` 负责“收紧”
- `DRW / EMA` 负责“别让少数类被淹没”

## 7. 评估

### 7.1 测试集评估入口

测试集评估入口是根目录的：

- `evaluate_test_set.py`

实际实现位于：

- `raman/eval/test_set_evaluator.py`

需要手动设置的核心参数通常包括：

- `EXP_DIR`
- `EVAL_LEVEL`
- `INHERIT_MISSING_LEVELS`
- `EVAL_ONLY_LEVEL`
- `EVAL_ONLY_PARENT`

注意

- 默认评估的是训练阶段保存下来的 `test_files.json` 对应样本，即`dataset_train/` 内部分出的 test split
- 而非 `dataset_test/` 目录里的外部测试集

### 7.2 评估流程

1. 从 `EXP_DIR` 读取 `config.yaml`、`hierarchy_meta.json`
2. 重建 `RamanDataset`
3. 读取训练阶段保存的 `train_files.json` / `test_files.json`
4. 按层级顺序逐层加载模型
5. 对每个样本做级联预测，直到目标层级
6. 汇总分类报告和混淆矩阵

如果实验目录下存在对应的 `*_prototypes.pt`，评估会按 `prototype_fusion_mode` 自动切换为分类头、prototype 或两者融合的打分方式；如果不存在，则自动回退到纯分类头结果。

当某一层是按父类拆开的子模型时，评估会自动：

- 先看上一层父类预测结果
- 再找到对应 parent 的子模型
- 如果该 parent 只有一个子类且训练时没有生成模型，则直接继承该唯一子类

这与真实预测流程保持一致

### 7.3 层级继承

这个开关用于控制：

- 如果样本在当前评估层级没有有效标签
- 是否回退到它实际存在的最深有效层级继续参与统计

当它为 `True` 时：

- 适合展示不完整层级下的整体级联效果
- 对多层训练但部分分支没继续下钻的情况更友好

当它为 `False` 时：

- 只有当前层真实有标签的样本才参与统计
- 指标更严格，也更“纯”

### 7.4 评估输出

`evaluate_test_set.py` 会在实验目录内生成：

```text
<EXP_DIR>/<EVAL_LEVEL>_test_result/
```

其中主要文件包括：

- `test_eval_results.csv`
- `classification_report.txt`
- `confusion_matrix_raw.csv`
- `confusion_matrix.png`

### 7.5 PCA + SVM 基线

传统基线入口是：

- `pca_svm_baseline.py`

实际实现位于：

- `raman/eval/baseline.py`

它和深度模型评估共用同一份训练/测试切分，因此可以做相对公平的对比。

当前流程是：

1. 从 `dataset_train/` 中按训练时的 split 提取 train/test 样本
2. 选择第一个输入通道，或把全部通道展平
3. `StandardScaler`
4. `PCA`
5. `SVM`
6. 输出准确率、分类报告、混淆矩阵和 PCA 散点图

输出目录为：

```text
<EXP_DIR>/<LEVEL>_baseline_test_result/
```

## 8. 分析

### 8.1 分析入口

分析入口保留在仓库根目录：

- `analyze.py`

实际实现位于：

- `raman/analysis/core.py`
- `raman/analysis/utils.py`

此外，根目录还提供了：

- `compare_test_train_means.py`

用于把外部测试菌与训练集中对应类、最近错误类的均值谱做直接对比，便于判断测试样本到底是模型没分开，还是光谱本身就更接近其他训练类。

入口内部通过 `ANALYSIS_MODE` 切换两种模式：

- `single`：分析单个全局模型，或某个 parent 对应的单个子模型
- `aggregate`：把多个 parent 子模型的结果按样本数加权聚合

### 8.2 单模型分析

单模型分析会围绕一个具体模型输出多种解释结果，包括：

- Integrated Gradients（IG）：用于解释输入（通道 & 波段）
- Layer Grad-CAM：用于评估各层的重要性

#### Integrated Gradients（IG）

对输入(x)，baseline(x')，目标函数 (F(x))
$$
\mathrm{IG}_i(x) = (x_i - x_i') \int_0^1 
\frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha
$$

- 从 baseline → 真实输入
- 沿路径累积梯度
- 得到每个输入维度的“累计贡献”

当前分析实现里：

- baseline 默认不是全零，而是从数据集中统计得到的平均光谱 `mean spectrum`
- 目标类别既可以选真实标签，也可以选模型预测类别
- 最终结果会对多个 batch 再做平均，因此比单次前向更稳定

如果某个峰在所有样本里都很强（平均光谱里已经有），那它在 IG 中可能贡献很小

所以 IG 更偏向发现区分性特征（discriminative features）而不是共有特征

在计算得到结果以后，IG的形状为`[B,C,L]`

对于输入通道重要性
$$
\mathrm{channel\space importance}_c= \mathrm{mean}_{b,l}(|\mathrm{IG}_{b,c,l}|)
$$

- 先对 attribution 取绝对值

  > 因为 attribution 可能有正有负，取绝对值后，强调的是“影响强度”，不是“方向”

- 再对 batch 维和波段维求平均

- 得到每个通道一个标量

对于类别波段重要性：

先沿通道维平均，得到每个样本一条长度为L的波段重要性曲线
$$
\mathrm{IG\_ band}_{b,l} = \mathrm{mean}_c (|\mathrm{IG}_{b,c,l}|)
$$
然后再按类别累加平均，最后得到：
$$
\mathrm{importance}[class,band]
$$
热图在回答：对某个类别而言，哪些波数位置最重要

#### Layer Grad-CAM

Layer Importance 不是传统二维图像里那种空间热图版本，而是把每一层压成一个标量重要性分数

实现方式是在待分析层上注册 forward / backward hook，分别保存：

- 前向输出 activation `A`
- 目标类别对该层输出的梯度 `G`

然后对每一层计算：

$$
\mathrm{Importance} = \mathrm{mean}|A\odot  G|
$$

- `A` 大，说明这一层在当前样本上激活强
- `G` 大，说明目标类别对这一层敏感
- 两者乘起来，再取绝对值平均，可以近似表示“这一层对当前决策有多重要”

当前实现会先在 block 级别算分，再做归一化

---

IG 用 (x - baseline) × grad是因为它在做路径积分，要衡量“从无到有”的贡献

Grad-CAM 用 A × grad是因为它在做局部线性近似，只看当前激活对输出的影响

### 8.3 聚合分析

聚合分析主要用于“某一层是按 parent 拆开训练”的场景

它会：

1. 逐个加载该层所有 parent 子模型
2. 分别计算每个子模型的波段重要性、通道重要性等结果
3. 再按样本数做加权聚合

聚合输出目录通常是：

```text
<EXP_DIR>/<analysis_level>_aggregate_analysis/
```

这种方式更适合回答：

- 在整一层上，哪些波段最稳定地重要
- 不同 parent 子模型的关注模式是否一致

### 8.4 目的

- 检查模型到底依赖了哪些波段和通道
- 比较不同层级、不同 parent 子模型的判别模式
- 观察 embedding 在训练集 / 测试集上的分布结构

对拉曼光谱任务来说，这些分析结果常常比单个 accuracy 更有解释价值，因为它能回答：

- 模型是学到了稳定峰位，还是学到了偶然噪声
- 不同类别的判别主要依赖局部峰，还是依赖组合关系
- 多层级拆模后，不同 parent 子模型的关注区域是否发生迁移

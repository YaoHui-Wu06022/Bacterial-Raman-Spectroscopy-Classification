# 拉曼光谱层级分类项目

## 1. 项目目标与任务定义

本项目面向细菌拉曼光谱识别任务，构建一套完整的层级分类实验系统

当前代码已经覆盖：

- 原始 `.arc_data` 的离线清洗与目录重组
- 面向 1D 光谱的在线预处理、多通道输入构建与数据增强
- 逐层级、按父类拆分的训练流程
- 独立测试集评估、传统机器学习基线对照
- 训练后可解释性分析与测试集 embedding 诊断

从任务定义上看，项目解决的是“层级分类”而不是“单层分类”

以 `细菌` 数据集为例，目录天然带有：

- 属级标签，例如 `Escherichia / Klebsiella / Proteus`
- 更细一级的叶级标签，例如 `EC / KP / PMI`

训练时，系统会先从目录结构自动构建层级标签树，再按当前训练层级决定本次训练的是：

- 顶层全局模型
- 或某一层下按父类拆开的子模型

因此，这个项目的真实任务可以概括为：

1. 把原始拉曼光谱统一到可训练的标准表示空间
2. 在该表示空间上学习稳定的层级判别边界
3. 在预测时按层级逐级细化，并对缺失子模型做兼容回退
4. 用分析工具判断模型到底在看哪些峰段、哪些层、哪些通道

项目当前的技术主线如下：

1. 离线预处理阶段：做 AsLS 基线校正、波段裁剪、坏段剔除、统一波数轴插值、训练集 PCA 异常值过滤
2. 在线输入阶段：从清洗后的单通道光谱构造模型输入，当前主线是 `SNV + smooth` 双通道
3. 模型阶段：使用多尺度 1D CNN 主干提取局部峰形，再接序列编码器和分类头
4. 训练阶段：围绕层级标签、类别不均衡和细粒度难样本设计多种损失与重加权策略
5. 评估与分析阶段：不仅看准确率，还通过混淆矩阵、embedding 近邻诊断、IG、Layer Grad-CAM 等方式分析错误来源

## 2. 仓库结构与模块职责

```text
拉曼光谱分类/
├─ train.py                          # 顶层训练入口，只负责传入当前训练层级和手动覆盖项
├─ evaluate_test_set.py              # 测试集评估入口
├─ pca_svm_baseline.py               # PCA+SVM 基线入口
├─ analyze.py                        # 统一分析入口（single / aggregate）
├─ compare_test_train_means.py       # 外部测试集 embedding 近邻诊断
├─ pack_raman.py                     # 打包 raman 库，便于上传到 Colab
├─ raman/
│  ├─ config.py                      # 训练配置定义：输入通道、模型结构、损失、增强、优化器参数
│  ├─ config_io.py                   # config.yaml 读写、实验目录重载
│  ├─ model.py                       # 主模型实现：多尺度 stem + 1D CNN + encoder + pooling + head
│  ├─ trainer.py                     # 训练主流程：建数据集、建模型、训练、验证、保存结果
│  ├─ data/
│  │  ├─ paths.py                    # 训练/测试目录解析，统一把 dataset_root 映射到具体阶段目录
│  │  ├─ dataset.py                  # 层级数据集扫描、标签编码、样本索引与 DataLoader 输入接口
│  │  └─ preprocess.py               # 在线预处理与增强：标准化、smooth/d1/raw 通道构建、训练增强
│  ├─ analysis/
│  │  ├─ core.py                     # 单模型/聚合分析调度：加载实验、选择目标模型、组织输出目录
│  │  └─ utils.py                    # IG、Layer Grad-CAM、embedding 提取与可视化等底层实现
│  ├─ eval/
│  │  ├─ experiment.py               # 实验目录解析、配置加载、层级名校验
│  │  ├─ report.py                   # classification report、混淆矩阵、文本结果输出
│  │  ├─ test_set_evaluator.py       # 测试集评估实现
│  │  └─ baseline.py                 # PCA+SVM 基线实现
│  └─ training/
│     ├─ split.py                    # 训练/验证切分、训练范围解析、父类过滤
│     ├─ eval.py                     # 训练期评估：按文件或层级计算指标、层级掩码推理
│     ├─ losses.py                   # Focal / SupCon / Center / class weight 等损失工具
│     ├─ session.py                  # 训练会话初始化：随机种子、输出目录、日志、配置快照
│     └─ sampler.py                  # 分层采样器
├─ dataset_process/
│  ├─ cli.py                         # 离线数据处理统一入口，负责 pack/classify/preprocess/count
│  ├─ profiles.py                    # 各数据集的目录布局、原始目录命名、统一坏波段设置
│  ├─ common.py                      # 离线预处理底层函数：读谱、AsLS、坏段掩码、单谱清洗、均值谱绘图
│  └─ pipeline.py                    # 离线主流程：打包、目录重组、训练集清洗、测试集清洗、统计
├─ predict/
│  ├─ predict_core.py                # 层级级联推理核心：逐层加载模型并向下细化预测
│  ├─ predict_folder.py              # 批量目录预测
│  └─ predict_single.py              # 单目录预测
├─ colab/
│  └─ colab_unified.ipynb            # Colab 一体化 notebook：解压库、数据处理、训练、评估、分析、打包
├─ notebooks/
│  └─ single_process_AsLS_cut_SNV.ipynb # 单条光谱从原始输入到模型通道构建的可视化 notebook
└─ dataset/
   ├─ 细菌/
   ├─ 耐药菌/
   ├─ 厌氧菌/
```

仓库采用“顶层薄入口 + 包内实现”的组织方式：

- 顶层 `train.py / evaluate_test_set.py / analyze.py` 只负责当前实验要改的参数
- 实现细节放在 `raman/` 和 `dataset_process/` 里
- Colab、本地脚本和后续自动化都能复用同一套包内逻辑，而不是复制大段 notebook 代码

## 3. 离线数据预处理

离线数据预处理统一走 `dataset_process`

常用命令：

- `pack-init`：把 `dataset_init/` 打成 `dataset_init.npz`
- `unpack-init`：把 `dataset_init.npz` 还原为目录结构
- `classify`：扫描 `dataset_init/` 或压缩包内容，按文件名前缀规则重组到 `dataset_train_raw/`
- `preview-init`：对原始数据做预览，不生成训练集，只输出均值谱图，方便先检查原始质量和类别分布
- `preprocess-train`：对 `dataset_train_raw/` 做完整训练集离线清洗，并额外执行 PCA 异常值过滤，输出到 `dataset_train/`
- `preprocess-test`：对 `dataset_test_raw/` 做与训练集一致的离线清洗，但不做 PCA 异常值剔除，输出到 `dataset_test/`
- `count`：统计指定目录下各层级类别的样本数量，用于检查重组结果、清洗结果和类别不均衡情况

### 3.1 参数修改位置

离线清洗参数不从 CLI 传入，统一在 `dataset_process/pipeline.py` 里修改：

- `DEFAULT_PIPELINE_CONFIG`

默认设置涵盖：

- 波段裁剪范围 `cut_min` / `cut_max`
- 统一参考波数轴点数 `target_points`
- AsLS 参数 `asls_lam` / `asls_p` / `asls_max_iter`
- 训练集最小样本数 `min_samples_per_class`
- 绘图归一化方式 `norm_method`
- PCA 异常值过滤相关参数

不同数据集的目录名在 `dataset_process/profiles.py` 里维护；坏波段现在全局固定为 `890~950 cm^-1`

### 3.2 输入目录与输出目录

离线流程主要使用这些目录：

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

如果某个分组预处理后样本数少于 `min_samples_per_class`，该分组会跳过

被 PCA 剔除的样本会记录到 `log.txt`

#### AsLS 基线校正原理

AsLS（Asymmetric Least Squares，非对称最小二乘）是一种通过“加权平滑 + 非对称惩罚”来估计光谱基线的方法：

- 用二阶差分约束基线的“平滑性”
- 用非对称权重抑制峰值对基线拟合的影响（峰值被当作“异常点”处理）
- 通过迭代更新权重，使基线逐步贴合“背景”而避开“信号峰”

目标是估计基线 $z$

优化问题为：
$$
\min_{z} \sum_{i=1}^{n} w_i (y_i - z_i)^2  +  \lambda \sum_{i=1}^{n-2} (z_{i+2} - 2z_{i+1} + z_i)^2
$$
矩阵形式为：
$$
\min_{z}  (y - z)^T W (y - z) + \lambda z^T D^T D z
$$
其中：

- $W = \mathrm{diag}(w_1, w_2, \dots, w_n)$：权重矩阵

  权重 $w_i$ 根据残差动态更新
  $$
  w_i =
  \begin{cases}
  p, & y_i > z_i \\
  1 - p, & y_i \le z_i
  \end{cases}
  $$
  $y_i > z_i$（可能是峰） → 权重小 → 不强制拟合

  $y_i \le z_i$（基线区域） → 权重大 → 强制贴合 

- $D$：二阶差分矩阵
  $$
  D =
  \begin{bmatrix}
  1 & -2 & 1 & 0 & \cdots \\
  0 & 1 & -2 & 1 & \cdots \\
  \vdots & & & & \ddots
  \end{bmatrix}
  $$
  二阶差分矩阵是用于惩罚基线的“弯曲度”，保证基线平滑

- $\lambda$：平滑参数（控制基线光滑程度）

对目标函数求导(对$z$求导)可得：
$$
(W + \lambda D^T D) z = W y
$$
由于权重 $w_i$ 依赖于基线估计 $z$，而 $z$ 又依赖于 $w_i$，因此该问题需要通过迭代求解

每次迭代包含两个步骤：

1. 在当前权重固定的情况下，解带权最小二乘问题，得到基线估计 $z$
2. 根据当前 $z$ 与观测值 $y$ 的关系，重新判断哪些点属于峰（赋小权重），哪些点属于基线（赋大权重）

通过不断重复“拟合 → 重新加权”的过程，使基线逐步向信号的下包络收敛

#### PCA 异常值过滤

PCA（Principal Component Analysis，主成分分析）是一种经典的线性降维方法，其核心思想是：

- 在原始高维空间中寻找一组新的正交基（主成分）
- 使得数据在这些方向上的投影方差最大
- 用前几个主成分尽可能保留原始数据的主要信息

对于光谱数据矩阵：
$$
X \in \mathbb{R}^{n \times p}
$$
对数据进行中心化：
$$
X_c = X - \bar{X}
$$
计算协方差矩阵：
$$
S = \frac{1}{n} X_c^TX_c
$$
对协方差矩阵做特征分解得到特征向量
$$
S = P\Lambda  P^T
$$

- $P$：特征向量(主成分方向)
- $\Lambda$：特征值(方差)

定义得分矩阵：
$$
T =  X_cP
$$

使用前 `k` 个主成分对该类别内每条光谱做重构

可以写出低维表示与重构结果：
$$
T_k = X_c P_k
$$
$$
\hat{X} = T_k P_k^T + \bar{X}
$$
对单个样本 $x_i$，使用重构均方误差作为异常度量：
$$
e_i = \frac{1}{p} \sum_{j=1}^{p} (x_{ij} - \hat{x}_{ij})^2
$$

按该类别内部误差分布取分位数阈值，每个类别删掉重构误差最高的 `3%` 样本
$$
\tau = Q_{1-r}(e)
$$
其中：

- $r$：异常值比例
- $Q_{1-r}(e)$：误差向量 $e$ 的 `1-r` 分位数

### 3.7 数据统计

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

### 4.1 训练数据处理流程

离线阶段完成以后，`dataset_train/` 和 `dataset_test/` 中保存的是“已经完成基线校正、坏段剔除、统一波数轴对齐”的单条光谱文本文件

`raman` 负责的不是再做一遍离线清洗，而是在训练期把这些单条光谱进一步转换成模型真正使用的输入张量

完整链路是：

1. `RamanDataset` 从 `dataset_train/` 扫描目录树
2. 自动构建层级标签树和类别映射
3. `__getitem__()` 读取单条 `.arc_data`，只取强度列
4. `build_model_input()` 按当前训练配置构造多通道输入
5. `DataLoader` 把单条样本堆叠成 batch
6. 模型最终接收 `[B, C, L]` 张量

`RamanDataset` 在扫描目录时会自动整理出：

- `level_1 ... level_N`
- `leaf`
- 每一层的 `label -> id` / `id -> label` 映射
- `parent_to_children`
- 每个样本对应的多层级标签编码

这样训练、评估、分析和预测都不需要手工维护类别表，而是直接依赖目录结构得到统一的层级语义

### 4.2 在线预处理与增强

`raman/data/preprocess.py` 当前已经回到旧版的“独立随机抽样 + 打乱 + 截断”增强模式，而不是配方驱动或双视图模式。

整体顺序是：

1. 读取离线清洗后的单条强度光谱
2. 如果 `augment=True`，先在 RAW 域做随机抽样增强，得到同一个 `mother_raw`
3. 从这条共享的 `mother_raw` 构造各输入通道
4. 主通道做标准化，并可继续叠加标准化后的弱形状增强
5. 将各支路显式堆叠成最终输入

这里依然区分两类增强：

- Stage A：RAW 域增强，模拟更接近物理采集层面的扰动
- Stage B：标准化后增强，模拟较弱的局部形状波动

#### Stage A：RAW 域增强

RAW 域增强发生在标准化之前，直接作用于原始强度光谱，主要模拟：

- 仪器噪声
- 批次差异
- baseline 残留
- 波数轴标定误差

当前 RAW 域会独立抽样这几类增强：

- `piecewise_gain`
- `noise`
- `baseline`
- `axis_warp`

这些增强的含义分别是：

- `piecewise_gain`：分段缩放峰高，模拟不同波段相对峰高比例变化
- `noise`：统一为强度相关高斯噪声，不再单独保留泊松分支
- `baseline`：弱 / 强两种残余背景扰动，最多只抽中一种
- `axis_warp`：模拟轻微的非刚性波数轴偏移

其中 `noise` 的形式为：

$$
\sigma(x) = a + b|x|
$$

也就是噪声标准差由“全局底噪”与“随信号强度变化的噪声项”共同决定。这样保留了原来想模拟的强度相关噪声特性，但不再单独分一条 Poisson 增强分支。

RAW 域增强的控制参数主要是：

- `p_piecewise_gain`
- `p_noise`
- `p_axis`
- `p_baseline_weak`
- `p_baseline_strong`
- `max_pre_augs`

实际执行时，这些增强先按各自概率独立抽样，再随机打乱顺序，最后只执行前 `max_pre_augs` 个

#### Stage B：标准化后增强

主通道在标准化之后，还会继续叠加一层较弱的形状扰动，用于模拟：

- 轻微峰位漂移
- 峰展宽
- 局部缺失或污染

当前标准化后增强也采用独立抽样，再打乱后截断的方式，保留三种：

- `shift`
- `broadening`
- `mask_attenuate`

对应的控制参数是：

- `p_shift`
- `p_broadening`
- `p_cut`
- `max_post_augs`

其中 `max_post_augs` 表示：即使抽中了多种标准化后增强，最终也只会执行打乱后的前若干个

### 4.3 模型输入

标准化后的单通道光谱不会直接送进模型，而是会按配置构造成多通道输入

所有通道共享同一个 RAW 增强后的母体光谱 `mother_raw`

- 先对原始输入在 RAW 域做增强，得到 `mother_raw`
- 再从 `mother_raw` 构造各个通道

当前各通道语义为：

- `base`：`mother_raw -> normalize -> 可选 post augment`
- `smooth`：`mother_raw -> SG smooth -> normalize`
- `d1`：`mother_raw -> SG smooth -> d1 -> normalize`
- `raw`：如果开启，则直接使用 `mother_raw`，不标准化

因此单条样本最终形状是：

```text
[C, L]
```

其中：

- `C` 是输入通道数，由配置决定

- `L` 是离线统一后的光谱长度

经过 `DataLoader` 后，模型实际接收的是：

```text
[B, C, L]
```

#### SG平滑

SG（Savitzky–Golay）平滑本质是：在滑动窗口内做局部多项式最小二乘拟合，再用拟合多项式在中心点的值替代原始值

在每个位置 $i$，取一个窗口$[i-m, \dots, i, \dots, i+m]$

在这个窗口内，用一个低阶多项式去拟合数据：
$$
y(i + k) \approx a_0 + a_1 k + a_2 k^2 + \cdots + a_d k^d
$$
其中：

- $i$：当前中心点
- $k$：相对偏移，$k \in [-m, m]$
- $d$：多项式阶数

在窗口内求解：
$$
\min_{a_0, \dots, a_d} \sum_{k=-m}^{m} \left( y(i+k) - \sum_{j=0}^{d} a_j k^j \right)^2
$$
进一步地，将窗口内数据写成向量形式：

$$
\mathbf y_i =
\begin{bmatrix}
y(i-m) \\
y(i-m+1) \\
\vdots \\
y(i+m)
\end{bmatrix}
$$

构造多项式拟合的设计矩阵：

$$
X =
\begin{bmatrix}
1 & (-m) & (-m)^2 & \cdots & (-m)^d \\
1 & (-m+1) & (-m+1)^2 & \cdots & (-m+1)^d \\
\vdots & \vdots & \vdots & & \vdots \\
1 & m & m^2 & \cdots & m^d
\end{bmatrix}
$$

则局部最小二乘问题可写为：

$$
\min_{\mathbf a} \|\mathbf y_i - X\mathbf a\|_2^2
$$

对$a$求导，其解析解为：

$$
\hat{\mathbf a} = (X^T X)^{-1} X^T \mathbf y_i
$$

由于平滑后的输出为中心点 $k=0$ 处的函数值，其实只需要关注$a_0$

对于$a_0$代换一下
$$
\mathbf{c}=\left[\begin{array}{llll}
1 & 0 & \cdots & 0
\end{array}\right]\left(X^{T} X\right)^{-1} X^{T}
$$
所以：
$$
\hat{y}(i) = \sum_{k=-m}^{m} c_k \, y(i+k)
$$
$c_k$只和$X$有关，$X$是根据窗口大小和阶数写死的，所以 $c_k$ 仅由窗口大小 $2m+1$ 和多项式阶数 $d$ 决定

因此，SG 平滑可等价表示为一个固定卷积核的线性滤波过程

在本项目中，SG 平滑不是作为离线清洗步骤使用，而是作为在线辅助通道构造的一部分

当前默认参数是：

- `win_smooth = 15`
- `polyorder = 3`

因此 `smooth` 通道的物理含义可以理解为：

- 在保持主峰整体形状的前提下，抑制高频抖动
- 给模型提供一个比主通道更稳定的峰形参考视图

`d1` 通道如果开启，则当前逻辑不是直接对原谱求导，而是：

1. 先对 RAW 增强后的信号做 SG 平滑
2. 再对平滑结果做一阶导
3. 最后再标准化

这样做的原因是，一阶导本身对局部噪声非常敏感，如果不先平滑，导数通道会更容易被高频噪声主导

### 4.4 训练集、验证集、测试集

- 训练入口使用的基础数据目录是 `dataset_train/`
- 训练集和验证集都是从 `dataset_train/` 内部分割得到
- 如果实验目录下已有 `train_files.json` 和 `test_files.json`，会优先复用原切分
- 如果没有，就按 `split_level` 重新分组切分

当前训练代码中：

- `train_dataset = RamanDataset(..., augment=True)`，用于训练
- `test_dataset = RamanDataset(..., augment=False)`，用于训练过程中的验证

`test_dataset` 其实是“验证集视角”，不是外部测试集

真正的独立测试集位于 `dataset_test/`，不参与训练期切分

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
- `C`：输入通道数，由 `smooth_use`、`raw_use`、`d1_use` 决定
- `L`：离线统一后的光谱长度

这套结构的设计目标不是做一个完全通用的 1D 分类器，而是围绕拉曼光谱的特点，把局部峰形建模、跨峰关系建模和最终判别头拆开，便于做消融实验

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

它不是训练期解释工具，而是独立测试集诊断脚本。当前这份脚本的主表示已经不是“均值谱对比”，而是：

- 测试谱 embedding 对训练谱 embedding 的最近邻诊断
- 测试文件夹 embedding centroid 对训练类 centroid 的相似度诊断
- 文件夹级模型投票结果汇总

脚本会读取：

- 实验目录中的模型权重
- `dataset_train/` 作为训练 embedding bank
- `dataset_test/` 作为独立测试样本来源

最终输出目录通常为：

```text
<EXP_DIR>/test_train_embedding_compare/
```

其中会包含：

- `summary.csv`
- 每个测试文件夹一张柱状图

#### 独立测试集 embedding 诊断图

每张图默认有两列柱状图：

- 左图 `Model Vote Top-K`
  - 表示该测试文件夹里的每条谱先过模型分类头
  - 每条谱取 `top1`
  - 再统计整个文件夹里各类别被投了多少票

- 右图 `Embedding Neighbor Vote Top-K`
  - 表示该测试文件夹里的每条谱先提取 embedding
  - 再在训练集 embedding bank 中找最近邻
  - 用最近训练样本的类别投票
  - 最后统计整个文件夹里各类别被投了多少票

图题格式是：

```text
<folder_name> | expected=<expected_label> | model_top1=<...> | neighbor_top1=<...> | centroid_top1=<...>
```

各字段含义分别是：

- `expected`：根据测试文件夹名推断出的理论对应类别
- `model_top1`：模型多数票第一名，也就是左图最高票类别
- `neighbor_top1`：embedding 最近邻多数票第一名，也就是右图最高票类别
- `centroid_top1`：先把整个测试文件夹的 embedding 求平均，再与各训练类 centroid 做余弦相似度，最接近的那个类别

这三者不一定一致，而这种不一致本身就是诊断信息：

- `model_top1` 错，但 `neighbor_top1` 对
  
  更像分类头没有把已有 embedding 信息用好
- `model_top1` 和 `neighbor_top1` 都错到同一类
  
  更像测试样本在 feature space 里本来就更贴近错误类
- `neighbor_top1` 对，但 `centroid_top1` 错
  
  更像文件夹内部是多峰结构，简单取平均把细节抹平了
- `centroid_top1` 对，但 `model_top1` 错
  
  更像整体中心仍然正确，但单谱层面边界样本较多

因此，这个脚本的用途不是单纯“再看一次预测结果”，而是帮助判断：

- 错误主要来自分类头
- 还是来自 embedding 空间本身
- 还是来自测试集与训练集之间的分布偏移

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

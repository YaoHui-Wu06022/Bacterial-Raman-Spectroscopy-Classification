# 拉曼光谱层级分类项目

## 1. 项目目标

本项目面向细菌拉曼光谱识别任务，构建一套完整的层级分类实验系统

当前代码已经覆盖：

- 原始 `.arc_data` 的离线清洗与目录重组
- 面向 1D 光谱的在线预处理、多通道输入构建与数据增强
- 逐层级、按父类拆分的训练流程
- 独立测试集评估、传统机器学习基线对照
- 训练后可解释性分析与测试集 embedding 诊断

项目解决的是“层级分类”，以 `细菌` 数据集为例，目录天然带有：

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
5. 评估与分析阶段：通过混淆矩阵、embedding 近邻诊断、IG、Layer Grad-CAM 等方式分析错误来源

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
- Colab、本地脚本和后续自动化都能复用同一套包内逻辑，无需复制大段 notebook 代码

## 3. 离线数据预处理

离线数据预处理统一走 `dataset_process`

常用命令：

```
python -m dataset_process pack-init 细菌     # 打包init数据集
python -m dataset_process unpack-init 细菌   # 还原init数据集
python -m dataset_process classify 细菌      # 扫描init数据集，按文件名前缀规则重组数据集
python -m dataset_process preview-init 细菌  # 对init每个文件夹做均值谱图输出，方便检查原始数据质量
python -m dataset_process preprocess-train 细菌 # 对训练数据进行清洗流程，并执行PCA清洗
python -m dataset_process preprocess-test 细菌  # 对测试数据进行一致的清洗
```

### 3.1 参数修改

离线清洗参数在 `dataset_process/pipeline.py` 里修改：

清洗参数集成为`DEFAULT_PIPELINE_CONFIG`

设置涵盖：

- 波段裁剪范围 `cut_min` / `cut_max`
- 统一参考波数轴点数 `target_points`
- AsLS 参数 `asls_lam` / `asls_p` / `asls_max_iter`
- 训练集最小样本数 `min_samples_per_class`
- 绘图归一化方式 `norm_method`
- PCA 异常值过滤相关参数

不同数据集的目录名在 `dataset_process/profiles.py` 里维护

由于设备采集差异，对所有数据集坏波段固定为`890~950 cm^-1`

### 3.2 数据集目录结构

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

### 3.3 原始数据预览

- 直接基于 `dataset_init/` 或 `dataset_init.npz` 做预处理预览
- 执行基线校正、裁剪、坏波段剔除与统一参考轴插值
- 不做 PCA 异常值过滤与光谱输出
- 只输出每个分组的均值谱图到 `dataset_init_fig/`

先检查原始数据质量，看是否需要将某个文件夹移除，不再进入后续训练数据集

### 3.4 重组数据集

由于采集数据按日期划分，原始数据集一般命名为`类别+数字`

所以需要重组数据，把多个文件夹按类别收缩到一个文件夹中

- 扫描 `dataset_init/` 或 `dataset_init.npz`

- 读取叶子目录名

- 统一按 `letters_sign` 规则提取类别前缀
  
  例如：`ABC12 -> ABC`，`ESBL+03 -> ESBL+`
  
- 将样本复制到 `dataset_train_raw/`，文件名更改为`叶子目录名_原文件名`

这一步的目的是先把原始采集目录整理成更稳定的类别目录结构，供后续统一清洗

### 3.5 训练集离线清洗

每条光谱执行：

1. 读取 `.arc_data`
2. AsLS 基线校正
3. 波段裁剪
4. 在裁剪后的原始波数轴上删除坏段
5. 对统一后的波数轴做线性插值，不跨坏段补点
6. 对同一分组样本按 PCA 重构误差做异常值过滤

如果某个分组预处理后样本数少于 `min_samples_per_class`，该分组会跳过

被 PCA 剔除的样本会记录到 `log.txt`

#### AsLS 基线校正原理

AsLS（Asymmetric Least Squares，非对称最小二乘）是一种常用的光谱基线估计方法，其核心思想是通过“平滑约束 + 非对称加权”拟合一条平滑背景曲线，并抑制信号峰对基线估计的干扰

包含两个关键机制：

- 使用二阶差分项约束基线的平滑性
- 使用非对称权重减弱峰上方数据点对拟合结果的影响

设原始光谱为 $y \in \mathbb{R}^n$，基线为 $z \in \mathbb{R}^n$，则 AsLS 的优化目标可写为

```math
\min_{z} \sum_{i=1}^{n} w_i (y_i - z_i)^2  +  \lambda \sum_{i=1}^{n-2} (z_{i+2} - 2z_{i+1} + z_i)^2
```

矩阵形式为

```math
\min_{z}  (y - z)^T W (y - z) + \lambda z^T D^T D z
```

其中：

- $W = \mathrm{diag}(w_1, w_2, \dots, w_n)$为对角权重矩阵

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
  \in \mathbb{R}^{(n-2)\times n}
  ```

  ```python
  D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length))
  ```

- $\lambda$为平滑参数，控制基线光滑程度

在权重固定时，对目标函数关于 $z$ 求导并令其为零，可得到线性方程组

```math
(W + \lambda D^T D) z = W y
```

```python
matrix_z = (matrix_w + lam * (D.T @ D)).tocsc()     # W + λD^T D
baseline = spsolve(matrix_z, weights * spectrum)    # 解 z
```

由于权重 $w_i$ 本身依赖于当前基线估计 $z$，因此 AsLS 需要通过迭代方式求解

权重更新规则：

```math
w_i =
\begin{cases}
p, & y_i > z_i \\
1-p, & y_i \le z_i
\end{cases}
```

- 当 $y_i > z_i$ 时，该点更可能位于峰上方，赋予较小权重
- 当 $y_i \le z_i$ 时，该点更可能属于背景区域，赋予较大权重

```python
weights = np.where(spectrum > baseline, p, 1 - p)
```

所以 AsLS 的求解过程其实就是不断重复两步：

1. 固定当前权重，求解基线
2. 根据新的基线更新权重

经过多次迭代后，基线会逐步贴近背景区域，同时避开主要信号峰

> AsLS vs airPLS / arPLS

#### PCA 异常值过滤

PCA（Principal Component Analysis，主成分分析）是一种经典的线性降维方法，其基本思想是在原始高维空间中寻找一组新的两两正交的坐标轴，使样本在这些方向上的投影方差依次最大

通过保留前几个主成分，可以在尽量保留主要信息的同时去除冗余噪声

对于某一类别内的光谱数据矩阵 

```math
X \in \mathbb{R}^{n \times p}
```

首先对数据按列中心化，得到

```math
X_c = X - \bar{X}
```

在中心化数据基础上，构造协方差矩阵

```math
S = \frac{1}{n} X_c^TX_c
```

对协方差矩阵进行特征分解，可得

```math
S = P\Lambda  P^T
```

- $P$ 为特征向量矩阵，其列向量对应主成分方向
- $\Lambda$ 为对角特征值矩阵，其对角元素表示各主成分所对应的方差大小

样本在主成分空间中的投影（得分矩阵）为

```math
T =  X_cP
```

在异常值剔除阶段，仅保留前$k$个主成分来表征该类别的主要结构

则对应的低维表示为

```math
T_k = X_c P_k
```

基于前$k$个主成分，可将样本重构回原始空间，得到重构结果

```math
\hat{X} = T_k P_k^T + \bar{X}
```

对单个样本 $x_i$，使用重构均方误差作为异常度量

```math
e_i = \frac{1}{p} \sum_{j=1}^{p} (x_{ij} - \hat{x}_{ij})^2
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

- 训练层级：`level_1 ... level_N`
- 每层的类别映射 `label -> id` / `id -> label` 
- 上下层级关系 `parent_to_children`
- 每个样本对应的多层级标签编码
- 供切分/分组使用的 `leaf` 内部标识

这样训练、评估、分析和预测都不需要手工维护类别表，而是直接依赖目录结构得到统一的层级语义

### 4.2 在线预处理与增强

整体顺序是：

1. 读取离线清洗后的单条强度光谱
2. 如果 `augment=True`，先在 RAW 域做随机抽样增强，得到共享的 `mother_raw`
3. 基于同一条 `mother_raw` 构造各个输入通道
4. 主通道做标准化，并可进一步叠加标准化后的弱形状增强
5. 将各支路显式堆叠成最终输入

当前实现将增强划分为两类：

- Stage A：RAW 域增强，用于模拟更接近采集过程或仪器层面的扰动；
- Stage B：标准化后增强，用于模拟幅度较小的局部形状变化

#### Stage A：RAW 域增强

RAW 域增强发生在标准化之前，直接作用于原始强度光谱，主要用于模拟以下扰动：

- 仪器噪声
- 批次差异
- baseline 残留
- 波数轴标定误差

当前 RAW 域会独立抽样这几类增强：

- `piecewise_gain`：分段缩放峰高，模拟不同波段相对峰高比例变化

- `noise`：高斯/泊松噪声

  `noise` 的形式为

  ```math
  \sigma(x) = a + b|x|
  ```

  噪声标准差由“全局底噪”与“随信号强度变化的噪声项”共同决定

- `baseline`：残余背景扰动

  分为强弱两类：

  - 弱 baseline 扰动：模拟幅度较小、形状较平滑的背景残留

    使用“线性趋势 + 低频正弦项”的组合来构造背景

    ```math
    b_{\text{weak}}(t) = \alpha t + \beta \sin(2\pi f t + \phi)
    ```

  - 强 baseline 扰动：模拟更明显的批次差异或仪器背景漂移

    不再限制为“线性 + 正弦”的简单形式，通过少量控制点在整条谱上构造一条分段平滑变化的低频曲线

    先在谱轴上随机放置若干个 knot，横坐标是等间距的，纵坐标先从对称均匀分布采样，再按样本幅值缩放

    ```python
    n_knots = np.random.randint(n_knots_min, n_knots_max + 1)
    xs = np.linspace(0, length - 1, n_knots, dtype=np.float32)
    ys = np.random.uniform(-1.0, 1.0, n_knots).astype(np.float32)
    ys *= np.random.uniform(amp_min, amp_max) * amp
    ```

    最后通过插值得到整条背景曲线

    ```math
    b_{\text{strong}}(t) = \mathrm{Interp}\big((t_1,y_1),\dots,(t_K,y_K)\big)
    ```

- `axis_warp`：模拟轻微的非刚性波数轴偏移

  先为原始坐标构造一个扰动后的坐标映射 $i' = i + \Delta(i)$

  $\Delta (i)$不是常数，随位置变化，所以不同波数位置的偏移量不同

  形变由两部分组成：一个线性项加一个正弦项

  ```math
  \Delta(i)=\alpha\,(i-c)+\beta \sin\!\left(2\pi \frac{i}{n}+\phi\right)
  ```

  不是直接改横轴保存，而是把原始光谱视为定义在扰动坐标上的信号，再插值回原始规则网格

  相当于坐标扭曲后再重采样

  ```math
  y(i)=x\!\bigl(i'(i)\bigr)
  ```

实际执行时，这些增强先按各自概率独立抽样，再随机打乱顺序，最后只执行前 `max_pre_augs` 个

#### Stage B：标准化后增强

主通道在完成标准化后，还可以进一步叠加较弱的形状扰动

当前标准化后增强保留以下三类操作：

- `shift`：轻微峰位漂移 $\tilde{x}[i] = x[i-s]$

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

这部分增强同样采用“独立采样—随机打乱—截断执行”的策略

即使一次采样命中了多种增强，最终实际执行的操作数仍然受 `max_post_augs` 限制

### 4.3 模型输入

标准化后的单通道光谱不会直接送进模型，而是会按配置构造成多通道输入

所有通道共享同一个 RAW 增强后的母体光谱 `mother_raw`，再基于这条共享谱线构造各个输入通道

当当前各通道的语义如下：

- `base`：`mother_raw -> normalize -> 可选 post augment`

- `smooth`：`mother_raw -> SG smooth -> normalize`

- `d1`：`mother_raw -> SG smooth -> d1 -> normalize`

  一阶导本身对局部噪声非常敏感，如果不先平滑，导数通道会更容易被高频噪声主导

- `raw`：若启用，则直接使用 `mother_raw`，不做标准化

对单条样本而言，最终输入张量的形状为：

```text
[C, L]
```

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

### 4.4 训练集、验证集、测试集

- 训练入口使用的基础数据目录是 `dataset_train/`
- 训练集与验证集都从 `dataset_train/` 内部划分得到
- 如果实验目录下已有 `train_files.json` 和 `test_files.json`，会优先复用原切分
- 如果没有，就按 `split_level` 重新分组切分

当前训练代码中：

- `train_dataset = RamanDataset(..., augment=True)`，用于训练
- `test_dataset = RamanDataset(..., augment=False)`，用于训练过程中的验证

这里的 `test_dataset` 只是训练阶段的验证集视图，并不等同于外部独立测试集

真正的独立测试集位于 `dataset_test/`，不参与训练期切分

## 5. 模型

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

这套结构的设计目标是围绕拉曼光谱的特点，把局部峰形建模、跨峰关系建模和最终判别头拆开，便于做消融实验

### 5.2 Backbone

前端特征提取器由 `backbone_type` 控制：

- `cnn`：使用 `RamanClassifier1D` 内部的 1D CNN 主干
- `identity`：跳过 CNN，仅做平均下采样和 `1x1` 通道投影

采用默认 `in_channels=2`

1. stem 输出 `64` 通道，并先做一次 `AvgPool1d(kernel_size=2)`  
2. `layer1`：`64 -> 64`
3. `layer2`：先池化再进入残差块，`64 -> 128`
4. `layer3`：先池化再进入残差块，`128 -> 256`
5. `layer4`：先池化再进入残差块，`256 -> 384`
6. 最后用 `1x1 Conv` 投影到统一的 `transformer_dim=192`

CNN 主干逐步扩大通道数，并通过四次时序压缩逐渐增大感受野

#### 5.2.1 多尺度 stem

stem 由 `stem_multiscale` 控制：

- `False`：单尺度 stem，结构为 `Conv1d + BN + Activation + AvgPool1d`
- `True`：多尺度 stem，默认开启

三条分支并联后，在通道维拼接成总计 `64` 个通道，再统一做一次平均池化  

这相当于在最前端同时观察三种局部尺度：

- 小卷积核更擅长捕捉尖锐峰、窄峰和局部突变
- 中等卷积核更适合提取相邻峰之间的组合关系
- 大卷积核更容易感知宽峰、缓变背景和峰包络

对于拉曼光谱，这比单一卷积核更自然，因为不同化学峰的宽度和局部形态本来就不一致

#### 5.2.2 残差 bottleneck 块

CNN 主干的基本单元是 `ResidualBottleneck1D`

每个 block 都由五部分组成：

1. `1x1 Conv` 降维  
2. `3x3 Conv` 做主卷积变换  
3. `1x1 Conv` 升维  
4. `SEBlock1D` 做通道重标定  
5. residual shortcut 与输出激活

一个 block 的数据流:
$$
x \rightarrow \underbrace{1\times1}_{reduce} \rightarrow \underbrace{3\times1}_{conv/group} \rightarrow \underbrace{1\times1}_{expand} \rightarrow SE \rightarrow +\;shortcut \rightarrow activation
$$
如果输入通道数和输出通道数不同，shortcut 会自动走 `1x1 Conv + BN` 投影

`conv_reduce`：调整通道数（降维），把原始特征压缩成一个更紧凑的表示

`conv_group`：ResNet为单个卷积核，Resnext为`cardinality`个卷积核，负责提取局部模式

`conv_expand`：调整通道数（升维），恢复通道数
$$
y = \phi(SE(F(x)) + x)
$$

#### 5.2.3 mid_channels

**ResNet 模式**
$$
\text{mid\_channels} = \max \left(\left\lfloor \frac{\text{out\_channels}}{\text{bottleneck\_ratio}} \right\rfloor, 1\right)
$$

当前默认 `resnet_bottleneck_ratio=4`

所以如果某个 stage 的输出通道是 `128`，那么中间 bottleneck 宽度就是 `32`

它的优点是结构直观、参数含义清楚，也更容易和经典 ResNet 论文中的 bottleneck 结构对应起来

**ResNeXt 模式**
$$
\text{mid\_channels} =
\max\left(
\underbrace{\text{out\_channels} \cdot \frac{\text{base\_width}}{64}}_{\text{每组的宽度}} \times \underbrace{\text{cardinality}}_{\text{组数}},
\text{cardinality}
\right)
$$

cardinality（最重要）：卷积多分支，分组

base_width：控制每个分支内部有多粗

当前默认：

- `cardinality = 4`
- `base_width = 4`

随后中间这层 `3x3 Conv1d` 会按 `groups=4` 分组计算，而不是全部通道一起进行卷积

结构一样，但每个 group 独立学习，权重完全不同
$$
F(x) = \sum_{i=1}^{G} T_i(x_i)
$$
ResNeXt 不是简单“把卷积做大”，而是把中间特征拆成多个并行子空间，再在 block 输出端重新融合

默认主线选 ResNeXt是因为对于拉曼光谱这种峰结构复杂、局部模式很多但每种模式本身又不一定很宽的信号，这种“分组表达 + 统一汇合”的方式往往比纯 ResNet 更有效

#### 5.2.4 SE 模块

每个 bottleneck 后面都可以接 `SEBlock1D`，由 `se_use` 控制，当前默认开启

1. 先对当前 block 输出做全局平均池化  
2. 通过两层全连接网络得到每个通道的缩放权重  
3. 再把这些权重乘回原始特征图

SE（Squeeze-and-Excitation）模块本质是在通道维度做注意力（channel attention）

- 自动学习“哪些通道更重要”
- 对重要通道加权放大，对不重要的抑制
- 是一种轻量级、可插拔的特征重标定机制

$$
X \in \mathbb{R}^{B \times C \times L}
$$

SE Block 一般分成两步：

1. Squeeze（压缩）
2. Excitation（激励 / 生成权重）

对每个通道，在长度维上做全局平均池化：
$$
z_{b,c} = \frac{1}{L} \sum_{i=1}^{L} X_{b,c,i}
$$
压缩成一个标量，作为这个通道的全局描述，也就是判断这个通道整体响应强不强

得到的通道描述向量输入一个小型 MLP，输出每个通道的权重(0-1)
$$
s = \sigma\left(W_2 \, \delta(W_1 z)\right)
$$
通常中间会先降维再升维
$$
\mathbb{R}^{C} \rightarrow \mathbb{R}^{C/r} \rightarrow \mathbb{R}^{C}
$$
主要有两个原因：

1. 减少参数量
2. 增加非线性建模能力

如果直接学一个$C\rightarrow C$ 的全连接，也能做，但参数更多，SE 采用瓶颈结构更经济

得到权重后再乘回原特征图
$$
\widetilde{X}_{b,c,i} = s_{b,c}  \odot X_{b,c,i}
$$
相当于对通道做抑制或增强操作

### 5.3 Encoder

前端 backbone 输出的是 `[B, C, L]` 形式的时序特征

通过特征投影转成 `[B, L, C]`，再交给序列编码器

`encoder_type` 支持三种模式：

- `transformer`
- `lstm`
- `none`

这一层的作用不是重新做全部局部峰提取，而是在 backbone 已经提炼出局部响应之后，继续建模不同波段之间的上下文关系

对拉曼光谱来说，可以理解为：

- backbone 更像“先找出哪些局部峰形有响应”
- encoder 更像“让峰 A 感知峰 B 是否同时出现，以及这些峰之间的组合关系”

#### transformer

当前实现使用的是一层轻量级 `TransformerEncoder`，输入维度来自前端 backbone 的投影输出

默认配置是：

- `transformer_dim = 192`
- `transformer_nhead = 6`
- `transformer_ffn_dim = 384`
- `transformer_layers = 1`
- `transformer_dropout = 0.2`
- `activation = "gelu"`
- `norm_first = True`
- `batch_first = True`

在进入 Transformer 之前，模型会先加上一层一维正余弦位置编码 `PositionalEncoding1D`

当前使用的是标准正余弦位置编码，最大支持长度为 `1000`

对位置 `pos` 和通道维度 `2i / 2i+1`，编码形式为：
$$
PE(pos, 2i) = \sin \left(pos \cdot 10000^{-2i / d_{\text{model}}}\right)
$$

$$
PE(pos, 2i+1) = \cos \left(pos \cdot 10000^{-2i / d_{\text{model}}}\right)
$$

前向时直接把位置编码和序列特征相加：

$$
X_{\text{pos}} = X + PE
$$

对于拉曼光谱，同样的局部形状如果出现在不同波数位置，含义可能完全不同，所以位置信息不能丢

当前使用的是 PyTorch 的 `nn.TransformerEncoderLayer` 再外包一层 `nn.TransformerEncoder`

单层结构可以概括成两部分：

1. 多头自注意力
2. 前馈网络 FFN

自注意力的核心形式是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

在多头机制下，输入会被投影到多个子空间分别做注意力，然后再拼接回来

FFN 则是作用在每个位置上的两层 MLP，它不负责跨位置交互，而是对每个位置已经融合好的上下文特征再做非线性变换

当前实现还启用了 `norm_first=True`，也就是先做 LayerNorm，再进入注意力和 FFN 子层

Post-LayerNorm是原文设计

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

Attention block：
$$
x_1 = \mathrm{LayerNorm}(x+\mathrm{Attention}(x))
$$
FFN block：
$$
x_2 = \mathrm{LayerNorm}(x_1+\mathrm{FFN}(x_1))
$$
可以这么理解
$$
y = \mathrm{LN}(x+F(x))\quad z= x+F(x)
$$
损失对输入的梯度：
$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial \mathrm{LN}}{\partial z}(I
+\frac{\partial F}{\partial x})
$$
问题在于LN在在残差连接之后，${\partial \mathrm{LN}}/{\partial z}$梯度值近似为$1/\sqrt d_k$，使得残差传播的恒等路径梯度不再是1，每层都要缩放一次

所以梯度随网络深度呈指数衰减，导致低层（靠近输入的层）梯度几乎消失，梯度消失会导致Adam等优化器的更新变得不稳定

Pre-LayerNorm 把 LayerNorm 移到子层前面

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

Attention block：
$$
x_1 =x+\mathrm{Attention}( \mathrm{LayerNorm}(x))
$$
FFN block：
$$
x_2 = x_1+\mathrm{FFN}(\mathrm{LayerNorm}(x_1))
$$
最关键的在于这样使得残差路径变成恒等映射
$$
y = x+F(\mathrm{LN}(x))\quad u = \mathrm{LN}(x)
$$
损失对输入的梯度：
$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y}(I+\frac{\partial F}{\partial u}\cdot \frac{\partial u}{\partial x})
$$
梯度中始终存在恒等梯度通路，梯度更稳定，深层 Transformer 更容易训练

#### lstm

当 `encoder_type="lstm"` 时，序列编码器会切换成 `nn.LSTM`

它和 Transformer 不同，不是靠自注意力一次性看全局，而是按序列顺序逐步更新隐藏状态，因此更接近经典时序建模方式

当前实现支持的 LSTM 配置包括：

- `lstm_hidden`
- `lstm_layers`
- `lstm_dropout`
- `lstm_bidirectional`

输入维度固定为前端投影后的 `proj_dim`，也就是当前默认的 `192`

如果打开双向 LSTM，那么最终序列维度会变成：
$$
\text{seq\_dim} = 2 \times \text{lstm\_hidden}
$$

否则就是：

$$
\text{seq\_dim} = \text{lstm\_hidden}
$$

LSTM 的核心是通过门控机制控制信息的保留、遗忘和输出，从而缓解普通 RNN 在长序列上容易出现的梯度消失问题。它包含三类门：

- 输入门（input gate）
- 遗忘门（forget gate）
- 输出门（output gate）

对时刻 `t`，常见写法可以表示为：

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

直觉上：

- 遗忘门决定旧记忆保留多少
- 输入门决定新信息写入多少
- 输出门决定当前时刻暴露多少隐藏状态

如果把 Transformer 理解成“显式建模任意两个位置之间的关系”，那么 LSTM 更像“沿着波数轴逐步积累上下文信息

但它的局限也很明确：

- 远距离位置之间的关系传播需要经过多步递推
- 并行性不如 Transformer
- 对“峰 A 和很远处峰 B 的直接组合关系”建模不如自注意力直接

因此，在本项目里 LSTM 更适合作为一个可对照的序列编码器基线

### 5.4 Pooling

encoder输出：
$$
X \in \mathbb{R}^{B \times L \times C}
$$
还需要把整条光谱压缩成一个固定长度的 embedding

怎么把 L 个位置的信息，合成一个向量，这就是 Pooling 的任务

`pooling_type` 支持：

- `attn`：注意力池化
  $$
  \alpha_t = \frac{e^{score_t}}{\sum_j e^{score_j}}\\
  z = \sum_{t=1}^{L} \alpha_t x_t
  $$
  模型自己学：哪些位置更重要

  但容易过拟合，且对数据量敏感

- `stat`：统计池化
  $$
  \mu = \frac{1}{L} \sum_{t=1}^{L} x_t\\
  \sigma = \sqrt{\frac{1}{L} \sum_{t=1}^{L}(x_t - \mu)^2}\\
  z = [\mu, \sigma]
  $$
  适合：类别差异是“整体分布”；噪声较多

### 5.5 Classifier

分类头由 `cosine_head` 控制：

- `True`：`CosineClassifier`

  先做 L2 归一化
  $$
  \hat{x} = \frac{x}{\|x\|_2}, \qquad \hat{w}_k = \frac{w_k}{\|w_k\|_2}
  $$
  归一化后内积就是余弦相似度
  $$
  z_k = s \cdot \cos(\theta_k)
  $$
  不看类别权重向量长度，只看类别中心方向

- `False`：普通 `Linear`
  $$
  z = Wx + b
  $$
  同时利用两种信息

  1. 特征和类别权重的方向是否一致
  2. 特征向量本身的模长大小

## 6. 训练

### 6.1 训练入口与实验目录

训练统一从根目录的 `train.py` 进入

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

其中 `hierarchy_meta.json` 记录了：

- 层级顺序
- 每层类别名
- `parent_to_children`
- 本次训练得到的全局模型和各 parent 子模型文件名
- 哪些父类因为只有一个子类而被直接记录为“确定映射”

后续预测、评估和分析都会复用这些元数据

当前训练运行时准备逻辑主要在 `raman/training/session.py`

它会负责：

- 设置随机种子和 cuDNN 可复现开关
- 创建输出目录与日志目录
- 把本次配置完整落盘到 `config.yaml`
- 同时写一份更适合人工复查的 `config_<timestamp>.txt`

日志层面现在分成两层：

- `run_<timestamp>.log`：整次训练的总日志
- `<model_tag>_<timestamp>.log`：具体某个模型的单独日志

这样做的好处是：

- 同一个实验目录里继续训练下一层时，不会覆盖之前的日志
- 回看某个 parent 子模型时，可以直接打开它自己的日志，而不必在总日志里手动定位

### 6.2 层级训练逻辑

训练入口里设置的是 `CURRENT_TRAIN_LEVEL`

- 数据集层级始终由 `dataset_train/` 目录树自动扫描得到
- `CURRENT_TRAIN_LEVEL` 只表示“这次训练实际要训练的那一层”

当 `train_per_parent=True` 时，训练行为是：

- 顶层没有父层，因此训练全局模型
- 对更细层级，如果某层有父层，就按父类拆成多个子模型分别训练
- 如果某个父类下只有一个子类，则不训练该 parent 子模型，只在元数据中记录这条确定关系

当前训练器并不是“自动从顶层一路往下跑完所有层级”，而是：

- 每次只训练一个 `CURRENT_TRAIN_LEVEL`
- 如果这一层启用了 `train_per_parent=True`，则在该层内部按父类拆成多个任务

如果当前实验目录缺少上一级模型或单子类记录，训练开始时会打印提示，提醒先训练哪一级

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

切分一旦生成并写入实验目录，后续继续训练同一实验目录下的更细层级时，会优先复用原来的 `train_files.json / test_files.json`

- 顶层模型和子模型尽量使用同一套训练/验证划分基准
- 多次补训练时，验证结果具有可比较性
- 不会因为每次重切分而造成实验波动

### 6.4 当前训练优化机制

当前训练的“优化”并不是只指优化器，而是四类同时生效的机制：

1. 参数更新侧：`Adam + weight_decay + CosineAnnealingLR`
2. 结构分区侧：不同模块使用不同学习率
3. 数据吞吐侧：`DataLoader` 预取与并行加载
4. 验证选择侧：用综合早停分数而不是单独看 `TestLoss`

#### 参数更新侧

当前训练器使用：

- `Adam`
- `weight_decay = 5e-4`
- `CosineAnnealingLR`

默认主学习率为：

```text
learning_rate = 4e-4
```

学习率调度参数为：

- `scheduler_Tmax = epochs`
- `scheduler_eta_min = 1e-5`

也就是说，主学习率会沿余弦曲线从 `4e-4` 逐步退火到 `1e-5`，训练前期更新更积极，后期更偏向细调和收敛。

#### 分组学习率

当前训练器不是对全模型使用统一学习率，而是按模块分组：

- 输入 stem：`0.6 × learning_rate`
- backbone 其他部分：`1.0 × learning_rate`
- 分类头：`1.2 × learning_rate`

这是一种“结构分区优化”，含义是：

- 输入 stem 更接近底层峰形提取，学习率略低，避免训练前期把基础局部结构扰乱
- backbone 主体保持标准学习率，承担主要表征学习
- 分类头学习率略高，便于更快贴合当前层级的类别边界

当前训练器对多尺度 stem 的参数分组也已经对齐，`stem_branches.*` 会与单尺度 stem 一样进入低学习率组。

#### DataLoader 设置

当前 DataLoader 相关默认配置来自 `config.py`：

- `train_loader_num_workers = 4`
- `eval_loader_num_workers = 4`
- `loader_pin_memory = True`
- `loader_persistent_workers = True`
- `loader_prefetch_factor = 2`

训练集 `DataLoader` 当前使用的是普通 `shuffle=True`，并没有额外引入自定义分层 batch sampler

因此这部分配置当前主要影响的是训练效率，而不是损失定义本身：

- 每个 epoch 的吞吐速度
- CPU 预取是否能跟上 GPU
- 验证阶段是否会被 I/O 明显拖慢
- batch 内正样本对的自然形成概率

最后这一点对 `SupCon Loss` 很重要，因为当前 `SupCon` 是在普通随机 batch 上计算的，它的有效性会受到 batch 内同类样本数量的影响

#### Early Stop

当前早停评分不是单纯看验证集准确率，也不是看最低 `TestLoss`，而是：

$$
\text{score} = w_{f1} \cdot \text{MacroF1} + w_{acc} \cdot \text{Accuracy}
$$

默认权重为：

- `early_stop_w_f1 = 0.6`
- `early_stop_w_acc = 0.4`

这意味着模型保存更偏向宏平均 F1，而不是只偏向头部类别占优时更容易好看的 Accuracy。

对当前这种类间不平衡、不同类别难度差异又较大的拉曼任务来说，这比“只看 Accuracy 保存 best model”更稳妥，因为：

- Accuracy 更容易被头部类主导
- Macro F1 更能反映尾部类和难类是否真正学到
- 两者加权后，可以避免模型选择过度偏向单一指标

还需要特别注意一条容易混淆的实现细节：

- 训练日志中的 `TrainLoss(cls)` 是主分类损失的 batch 平均
- `AlignLossW` 和 `SupConLossW` 是已经乘过当前 epoch 权重后的辅助损失
- 验证日志里的 `TestLoss` 来自验证阶段单独计算的 `CrossEntropyLoss`

因此：

- `TestLoss` 不是训练时反向传播所用的总损失
- best model 也不是按最低 `TestLoss` 保存
- 当前真正用于 model selection 的是上面的综合 `score`

### 6.5 当前训练总损失

如果按真实实现来写，当前训练总损失更接近下面两层结构：

总损失层：

$$
L_{\text{total}}(t)=L_{\text{primary}}(t)+\lambda_{\text{align}}(t)L_{\text{align}}+\lambda_{\text{supcon}}(t)L_{\text{supcon}}
$$

主损失层：

$$
L_{\text{primary}}(t)=\frac{1}{N}\sum_i \Big( s_i \cdot \text{Focal}(logits_i,y_i;w_{\text{dyn}}(t)) \Big)
$$

其中：

- `L_align`：当前层级的类内紧凑约束，对应 `hierarchical_center_loss`
- `L_supcon`：监督式对比损失
- $w_{\text{dyn}}(t)$：按 epoch 动态变化的类别权重
- $s_i$：样本级错误严重程度权重

当前代码里的 `L_primary` 并不是“单一 FocalLoss”，而是：

- `FocalLoss`
- `class_weights`
- `DRW / EMA dynamic weight`
- `severity weight`

共同叠加后的主分类损失

从这个角度看，当前训练的主损失已经同时覆盖了四个层面：

- 概率层：让真实类概率变高
- 类别层：避免少数类被头部类淹没
- 样本层：突出高置信错判等更危险错误
- embedding 层：通过 `align` 和 `supcon` 继续整理特征空间

#### Focal Loss

在光谱层级分类任务中，不同样本难度差异较大，容易样本会主导梯度，导致模型忽略难样本。

Focal Loss 在 CrossEntropy 的基础上增加一个可调节因子，抑制易样本梯度，放大难样本梯度，从而让训练更关注难样本。

`CrossEntropy Loss`：

$$
CE(p_t) = - \log(p_t)
$$

`Focal Loss`：

$$
FL(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- $p_t$：模型对真实类别的预测概率
- $\gamma$：控制对易样本的抑制程度
- $\alpha_t$：类别权重，当前实现中可由静态 `class_weights` 和动态 `DRW / EMA` 共同决定

当前实现对应：

```python
criterion = FocalLoss(
    gamma=config.gamma,               # 0.8
    weight=dynamic_weights,           # class_weights 经过 DRW / EMA 调整后的结果
    ignore_index=-1,
    label_smoothing=config.label_smoothing
)
```

这里还有两个实现细节值得说明：

- 当前 `gamma = 0.8`，属于偏温和的设置，不是极端强调 hard sample
- 当前项目并不是把所有“困难样本压力”都压在 Focal 上，而是让它和 `severity weight`、`DRW` 分工

也就是说：

- `FocalLoss` 主要处理“这个样本本身难不难”
- `DRW / EMA` 主要处理“这个类别是不是持续更难学”
- `severity weight` 主要处理“这次错误是不是特别危险”

#### class_weights

在进入动态重加权之前，训练器会先根据当前训练层的标签分布构造基础类别权重 `class_weights`

当前实现不是简单的反频率，而是使用对数平滑：

1. 统计当前训练层每个类别的样本数
2. 对计数做下界保护，避免出现 0
3. 按下式计算基础权重：

   $$
   \text{weight}_g = \frac{1}{\log(\text{count}_g + 1.5)}
   $$

4. 再归一化到平均值为 1：

   $$
   \text{weight}_g \leftarrow \frac{\text{weight}_g}{\frac{1}{C} \sum_{i=1}^{C} \text{weight}_i}
   $$

这样做的目的，是在照顾少数类的同时避免极端长尾下权重过大，导致训练振荡

可以把它理解成“静态类别不平衡校正”

#### DRW / EMA class weight

在训练过程中，类别难度并不是固定的

某些类别虽然样本数不算最少，但可能持续更难学

当前实现会对每个类别维护一条基于 CrossEntropy 的 EMA 难度轨迹，并据此动态调整类别权重

1. 对每个类别统计当前 batch 内的平均 `CrossEntropy`
2. 用 EMA 平滑历史难度：

   $$
   \text{EMA}_g(t) = \alpha \cdot \text{EMA}_g(t-1) + (1-\alpha) \cdot \text{CE}_g^{\text{batch}}
   $$

3. 计算相对难度：

   $$
   \text{raw\_diff}_g = \frac{\text{EMA}_g(t)}{\frac{1}{C} \sum_{i=1}^{C} \text{EMA}_i(t)}
   $$

4. 用相对难度修正基础类别权重：

   $$
   \text{weight}_g = \text{class\_weight}_g \cdot \Big(1 + \lambda (\text{raw\_diff}_g - 1)\Big)
   $$

5. 再归一化到平均值为 1

当前代码里的关键超参数是：

- `drw_start_epoch = 10`
- `ema_momentum = 0.9`
- `lambda_diff = 0.3`

#### severity weight

除了类别层面的重加权，当前训练器还会在样本层面再做一次“错误严重程度”加权

核心思路是：

- 如果模型虽然没把真实类排到第一，但已经排到 top-2 或 top-3，说明它并不是完全错离谱
- 如果模型以很高置信度把样本分错，说明这是一种更危险的错误，应当放大该样本在主损失中的贡献

当前实现里，会对每个有效样本：

1. 计算 `softmax(logits)`
2. 取前 `k=min(3, num_classes)` 个预测类别
3. 统计真实类别在 top-k 中的排名
4. 根据排名和 top-1 置信度给样本乘一个额外权重

当前规则按类别数自适应：

- 二分类：
  - 不额外使用 `severity weight`
  - 权重固定为 `1.0`
- 三分类：
  - 若真实类别排在 top-2，样本权重为 `0.90`
  - 若高置信度错判且真实类别排在 top-2，权重升到 `1.10`
  - 若高置信度错判且真实类别落到 rank-3，权重升到 `1.45`
- 四类及以上：
  - 若真实类别排在 top-2，样本权重为 `0.85`
  - 若真实类别排在 top-3，样本权重为 `0.95`
  - 若高置信度错判且真实类别排在 top-2，权重升到 `1.20`
  - 若高置信度错判且真实类别排在 rank-3 或更后，权重升到 `1.80`

其中高置信度阈值也按类别数调整：

- 三分类使用 `0.85`
- 四类及以上使用 `0.80`

因此这部分更准确地说，是一种“按预测错误严重程度调节主损失”的策略：

- 降低“几乎分对”的样本对梯度预算的占用
- 提高“高置信度错判”样本的学习强度
- 与 `FocalLoss` 形成互补：`FocalLoss` 更关注概率难度，`severity weight` 更关注错误排序结构

#### SupCon Loss

当前实现里的 `SupCon Loss` 是单视角监督式对比损失，不是双视图对比学习

`SupCon` 是直接在当前 batch 的 `feat` 上做同类拉近、异类推远

核心思想：

- 在 embedding 空间约束样本的相对距离
- 同类样本靠近
- 不同类样本远离
- 允许同一类别内部存在多个簇，因此它不强制类内单中心

公式与实现过程为：

1. 对 embedding 做 L2 归一化：

   $$
   z_i = \frac{feat_i}{||feat_i||_2}
   $$

2. 计算两两相似度矩阵：

   $$
   sim(i,j) = \frac{z_i \cdot z_j}{\tau}
   $$

3. 只把“同类且不是自己”的样本当作正样本对
4. 做数值稳定化
5. 计算每个 anchor 相对于所有候选样本的 log-prob
6. 对每个 anchor 平均其全部正样本

写成公式即：

$$
\log p_{ij} = sim(i,j) - \log\sum_{k \neq i} e^{sim(i,k)}
$$

$$
L_i = - \frac{1}{|P(i)|} \sum_{p \in P(i)} \log p_{ip}
$$

$$
L_{\text{supcon}} = \frac{1}{B} \sum_i L_i
$$

- 当前 batch 来自普通随机 `shuffle`，没有真正启用专门为对比学习设计的 batch sampler
- 因此 `SupCon` 的有效性依赖于 batch 内是否自然出现足够多的同类样本对

#### Center Loss

当前代码中的 `hierarchical_center_loss` 更接近一种 batch 内经验中心约束，做法是：

1. 在当前 batch 中取出某一类别的全部有效样本 embedding
2. 用这些样本的均值作为该类别在当前 batch 内的经验中心
3. 计算该类别样本到这个经验中心的平方距离
4. 对当前 batch 内所有有效类别的类内方差做平均

如果记当前 batch 内类别 $g$ 的样本集合为 $S_g$，则经验中心可写为：

$$
c_g^{(\text{batch})} = \frac{1}{|S_g|}\sum_{i \in S_g} x_i
$$

对应的类内紧凑约束更接近：

$$
L_{\text{align}} =
\frac{1}{|G_{\text{valid}}|}
\sum_{g \in G_{\text{valid}}}
\left(
\frac{1}{|S_g|}
\sum_{i \in S_g}
||x_i - c_g^{(\text{batch})}||_2^2
\right)
$$

在本项目中的实际使用方式是：

- 实现上对应的是 `hierarchical_center_loss`
- 当前训练某一层模型时，只对当前层级施加这一项约束

| 特性              | SupCon Loss                         | Center / Align Loss      |
| ----------------- | ----------------------------------- | ------------------------------------- |
| 类内约束          | 同类靠近（允许多簇）                | 让 batch 内同类更紧凑                 |
| 类间约束          | 间接推远不同类                      | 不直接约束                            |
| 对 batch 大小敏感 | 高（依赖足够正样本对）              | 中等（依赖 batch 内同类样本数）       |
| 中心定义          | 不显式引入类中心                    | 使用 batch 内经验中心                 |
| 使用场景          | 表征学习、相对距离整理、多模态更友好 | 分类增强、类内紧凑化                  |

| Loss                     | 约束目标                 | 作用对象                         |
| ------------------------ | ------------------------ | -------------------------------- |
| FocalLoss                | 分类概率与样本难度       | 样本层面                         |
| class_weights            | 静态类别不平衡校正       | 类别层面                         |
| DRW / EMA                | 动态类别难度校正         | 类别层面                         |
| severity weight          | 错误严重程度重加权       | 样本层面                         |
| SupCon Loss              | 类内靠近、类间分离       | embedding 层面，相对距离约束     |
| Center / Align Loss      | 当前层类内紧凑           | embedding 层面，batch 内方差约束 |

### 6.6 辅助损失调度原理

`align` 和 `supcon` 这两个辅助损失并不是从 epoch 1 就全权打开，而是按时间过程逐步介入

当前训练中的 epoch 权重更接近下面的分段逻辑：

$$
\lambda_{\text{align}}(t)=
\text{LinearRamp}(t;\text{align\_start},\text{align\_end},0,\lambda_{\text{align}}^{\max})
\times d(t)
$$

$$
\lambda_{\text{supcon}}(t)=
\text{LinearRamp}(t;\text{supcon\_start},\text{supcon\_end},0,\lambda_{\text{supcon}}^{\max})
\times d(t)
$$

其中：

- `align_start = 20`
- `align_end = 50`
- `supcon_start = 30`
- `supcon_end = 50`
- `decay_start_ratio = 0.7`

后期衰减因子 $d(t)$ 的逻辑是：

- 当 `t <= decay_start_ratio * epochs` 时，`d(t)=1`
- 当 `t > decay_start_ratio * epochs` 时，开始线性下降
- 当前实现给它设置了下界，因此最终不会降到 `0`，而是至少保留 `20%` 权重

如果把训练分阶段理解，可以近似看成：

1. 前期：
   - 主要由 `L_primary` 学分类边界
   - `align` 和 `supcon` 权重接近 0 或仍在启动
2. 中期：
   - 主分类损失继续工作
   - `align` 与 `supcon` 逐步进入稳定权重区间
   - embedding 的几何结构开始被更强地整理
3. 后期：
   - 辅助损失共同衰减
   - 让优化重点重新回到分类边界细调，而不是持续强拉 embedding

这样设计的考虑是：

- 前期先把“分对”这件事学稳
- 中期再加强特征空间结构约束
- 后期避免辅助损失持续过强，妨碍最终分类头收敛

### 6.7 训练损失之间的分工

从训练行为的角度看，当前项目不是简单把多个 loss 堆在一起，而是在四个层面共同作用于同一次反向传播：

1. 概率层：
   - `FocalLoss`
   - 负责把真实类概率推高，把分类边界学出来

2. 类别层：
   - `class_weights`
   - `DRW / EMA`
   - 负责决定“哪些类别不该在训练中被淹没”

3. 样本层：
   - `severity weight`
   - 负责决定“哪些具体错误更危险、更值得放大梯度”

4. embedding 层：
   - `SupCon Loss`
   - `Center / Align Loss`
   - 负责整理分类头之前的特征空间几何结构

因此它们不是重复堆料，而是分工不同：

- `FocalLoss` 解决“分错”和“难样本”
- `class_weights + DRW` 解决“哪些类不该被训练过程淹没”
- `severity weight` 解决“哪些错误更危险、更值得纠正”
- `SupCon` 解决“相对距离结构”
- `Center / Align` 解决“当前层类内紧凑性”

如果把它们对应回总损失：

- `L_primary` 负责主判别目标
- `L_primary` 内部已经包含类别层和样本层的重加权
- `L_align` 与 `L_supcon` 负责 embedding 几何约束

所以当前训练设计的闭环可以概括为：

- 用 `FocalLoss` 学分类边界
- 用 `class_weights + DRW` 修正类别不平衡
- 用 `severity weight` 提高高置信错判的学习强度
- 用 `SupCon` 拉开 embedding 的相对结构
- 用 `Center / Align` 收紧当前层的类内分布

## 7. 评估

### 7.1 测试集评估

深度模型评估入口在仓库根目录：

- `evaluate_test_set.py`

通常只需要手动设置这几个参数：

- `EXP_DIR`
- `EVAL_LEVEL`
- `INHERIT_MISSING_LEVELS`
- `EVAL_ONLY_LEVEL`
- `EVAL_ONLY_PARENT`

评估器评估的是训练期已经保存下来的 `test_files.json` 对应样本，不是 `dataset_test/` 外部独立测试集评估入口

如果实验目录里缺少 `train_files.json / test_files.json`，评估器会按训练配置里的 `split_level`、`train_split` 和 `seed` 重新构造 test split，再继续评估

### 7.2 评估流程

主流程可以概括为：

1. 从 `EXP_DIR` 加载 `config.yaml` 和 `hierarchy_meta.json`
2. 把 `dataset_root` 重新对齐到训练阶段目录，重建 `RamanDataset(augment=False)`
3. 解析目标评估层级 `eval_level`，得到从顶层到目标层的 `level_order`
4. 加载这条层级链路上需要的模型
5. 对每个测试样本逐层向下预测，直到目标层级
6. 把预测结果映射到当前评估的展示类别空间
7. 汇总准确率、Macro F1、Macro Recall、分类报告和混淆矩阵

这里有两个实现细节很重要：

- 评估器优先复用 `hierarchy_meta.json` 里记录的模型路径和父子关系
- 如果某一层是按父类拆开的子模型，评估时会先得到上一层父类预测，再决定进入哪个 parent 子模型

对于按父类拆开的层，级联逻辑是：

- 先用上一级模型得到父类预测
- 再找到该父类对应的子模型
- 如果该父类只有一个子类且训练时没有单独保存模型，就直接继承这个唯一子类

这和真实预测流程保持一致，而不是把每一层都当成互不相关的独立分类任务

### 7.3 统计口径开关

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

开启继承后，报告里的类别顺序可能会出现“保留父类”和“下钻到子类”混合的显示类名

这是评估器为了兼容未下钻分支而构建的展示空间，不等同于简单读取某一层的全部原始类名

`EVAL_ONLY_PARENT`：这个开关用于把评估范围限制到某个父类分支下

打开以后，评估器会：

- 只保留该 parent 对应的子模型
- 只统计这个父类内部允许出现的类别
- 自动过滤掉不属于这个 parent 的测试样本

这在两种场景下尤其有用：

- 检查某个父类内部的细分类效果
- 诊断“顶层分对了以后，子层到底分得怎么样”

### 7.4 评估输出

`evaluate_test_set.py` 会在实验目录内生成：

```text
<EXP_DIR>/<EVAL_LEVEL>_test_result/
```

主要文件和用途如下：

- `test_eval_results.csv`
  - 逐样本结果表
  - 保存样本路径、真实标签和预测标签
  - 当前保存的是映射后的整型标签，适合后续再做二次统计或错误样本回查
- `classification_report.txt`
  - 统一格式的分类报告
  - 包含每类的 `precision / recall / f1-score / support`
  - 同时包含 `accuracy`、`macro avg` 和 `weighted avg`
- `confusion_matrix_raw.csv`
  - 原始混淆矩阵计数表
  - 适合导出到表格工具里进一步分析
- `confusion_matrix.png`
  - 混淆矩阵热图
  - 当前图中是按行归一化的百分比，并同时标出原始计数

评估器在终端主摘要里会明确打印：

- `Accuracy`
- `Macro F1-score`
- `Macro Recall`

其中更推荐优先看：

- `Accuracy`：整体判对率
- `Macro F1-score`：类别均衡视角下的综合分类表现
- `Macro Recall`：各类召回是否均衡，是否有某些类被系统性漏判

而 `classification_report.txt` 里的 `weighted avg` 更适合辅助判断头部类是否明显拉高了整体表现

### 7.5 PCA + SVM 基线

除了深度模型级联评估，仓库里还保留了一条传统机器学习基线作为对照：

- `pca_svm_baseline.py`

实际实现位于：

- `raman/eval/baseline.py`

它和深度模型评估共用同一份训练/测试切分，因此可以做相对公平的对比。

当前流程很直接：

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

其中最常用的结果文件是：

- `metrics.txt`
  - 记录 Accuracy、PCA 保留维数、解释方差比例和分类报告
- `confusion_matrix.png`
  - 基线模型的混淆矩阵热图
- `pca_scatter.png`
  - 训练集在 PCA 前两维上的散点图，适合快速观察可分性

这一小节的意义不是替代深度模型，而是给当前实验一个低复杂度、可解释、可复现的传统基线参照

## 8. 分析

### 8.1 分析入口与两种模式

分析入口保留在仓库根目录：

- `analyze.py`

实际实现位于：

- `raman/analysis/core.py`
- `raman/analysis/utils.py`

当前分析主线由 `run_analysis_pipeline(...)` 统一调度，支持两种模式：

- `single`
  - 对一个具体模型做完整分析
  - 如果当前层没有全局模型，也可以自动退化为对多个 `parent` 子模型分别分析
- `aggregate`
  - 面向“该层按 parent 拆模训练”的场景
  - 逐个加载所有 `parent` 子模型，分别计算解释结果，再做加权聚合

从输出结果上看，这两种模式回答的问题并不相同：

- `single` 更适合回答“这个具体模型到底在看什么”
- `aggregate` 更适合回答“这一层整体上稳定依赖哪些模式”

### 8.2 单模型分析会输出什么

单模型分析会围绕一个具体模型输出多种解释结果，核心包括：

- 输入通道重要性 `channel_importance_IG.png`
- 各类别波段重要性热图 `band_importance_heatmap.png`
- 每类最重要波段表 `band_topK_per_class.csv`
- 各层或各 stage 的重要性图 `layer_importance.png`
- embedding 可视化图

这些结果并不在回答同一个问题：

- 输入通道重要性回答“模型更依赖哪一种输入表示”
- 波段热图回答“对某个类别而言，哪些波数位置最能驱动判别”
- 层重要性回答“模型决策更依赖前段特征提取，还是后段高层表征”
- embedding 图回答“不同类别在特征空间里是否真的被拉开”

### 8.3 Integrated Gradients（IG）

Integrated Gradients 用于解释输入维度对目标输出的贡献。对输入 $x$、baseline $x'$ 和目标函数 $F$，其定义为：

$$
\mathrm{IG}_i(x)= (x_i-x'_i)\int_{0}^{1}
\frac{\partial F(x' + \alpha (x-x'))}{\partial x_i}\, d\alpha
$$

它的核心思想是：

- 不直接看输入点 $x$ 处的一次梯度
- 而是从 baseline 逐步走到真实输入
- 在整条路径上累计梯度
- 最终得到每个输入维度的累计贡献

这样做的原因是，单次梯度容易受局部非线性影响，而路径积分更接近“从无到有”地衡量一个特征对输出的贡献。

当前仓库中的实现有几个重要细节：

- baseline 默认不是全零，而是由前若干个 batch 统计得到的平均光谱 `mean spectrum`
- 目标类别可以选择真实标签，也可以选择模型预测类别
- 实现上会用离散的 `alpha` 采样去近似上式中的积分
- 最终会对多个 batch 的结果再做平均，降低单批样本波动

其中 baseline 选平均谱而不是零谱，背后的考虑是：零输入在数学上可行，但对拉曼光谱并不总是物理合理；平均谱通常更接近真实数据流形，因此解释结果更稳定，也更容易和实验直觉对应

如果某个峰在所有类别中都普遍存在，那么它在平均谱里往往已经被“抵消”了一部分，这会让它在 IG 中显得不那么重要

因此 IG 更偏向发现区分性特征，而不是所有样本共有的强峰

在本项目里，IG 首先得到的是形状为 `[B, C, L]` 的 attribution，其中：

- `B` 是 batch 大小
- `C` 是输入通道数
- `L` 是波段长度

对于输入通道重要性，会先取 attribution 绝对值，再沿 batch 维和波段维求平均：

$$
\mathrm{ChannelImportance}_c
= \mathrm{mean}_{b,l}\left(|\mathrm{IG}_{b,c,l}|\right)
$$

这样得到的每个通道都是一个标量，强调的是该通道对判别的影响强度，而不是正向还是负向影响

对于类别波段重要性，仓库会先沿通道维聚合，得到每个样本的一条波段重要性曲线：

$$
\mathrm{BandIG}_{b,l}
= \mathrm{mean}_{c}\left(|\mathrm{IG}_{b,c,l}|\right)
$$

再按类别累加平均，得到最终热图：

$$
\mathrm{Importance}[class, band]
$$

这张热图回答的是：对某个类别而言，哪些波数位置对当前模型的决策最关键

### 8.4 Layer Grad-CAM

不是图像任务里常见的二维空间热图版本，而是用于评估中间层重要性的 layer-wise importance 分析

基本思想是：如果某一层在当前样本上激活很强，同时目标类别对这层输出又非常敏感，那么这一层就更可能对最终决策起到了关键作用

当前实现会在待分析层上注册 forward / backward hook，分别保存：

- 前向输出 activation `A`
- 目标类别对该层输出的梯度 `G`

然后对每一层计算：

$$
\mathrm{Importance}=\mathrm{mean}\left(|A \odot G|\right)
$$

这个式子可以这样理解：

- `A` 大，表示该层在当前输入上确实被强烈激活
- `G` 大，表示目标类别对该层变化非常敏感
- $A \odot G$ 同时把“激活强度”和“决策敏感性”纳入考虑
- 再取绝对值并做平均，就得到一个可比较的层重要性标量

这里要特别区分 IG 和 Layer Grad-CAM：

- IG 解释的是输入维度，从 baseline 到输入的路径贡献
- Layer Grad-CAM 解释的是中间层响应，只看当前激活和当前梯度

当前仓库还会把多个 block 的分数合并到更高一级的 stage 粒度，例如：

- `conv1`
- `layer1`
- `layer2`
- `layer3`
- `layer4`
- `transformer`
- `lstm`

因此最终图中的分数更适合做结构层面的解释，而不是精细到某个单独神经元的解释

### 8.5 聚合分析

聚合分析主要用于某一层是按 `parent` 拆开训练的场景，例如一个 level 下有多个 `parent` 子模型

它的基本流程是：

1. 逐个加载该层所有 `parent` 子模型
2. 分别计算每个子模型的通道重要性、层重要性和波段重要性
3. 再把这些结果按样本数或类别计数进行加权聚合

聚合输出目录通常是：

```text
<EXP_DIR>/<analysis_level>_aggregate_analysis/
```

聚合结果与单模型结果的区别在于，它不是在描述某一个具体子模型，而是在描述“这一层整体上最稳定的判别模式”

在当前实现中：

- 输入通道重要性会按各 `parent` 训练样本数加权
- 层重要性会先合并到 stage 粒度，再按样本数加权
- 类别波段热图会按每类实际参与统计的样本数加权
- 某些缺失类还会尝试从上一级 parent 模型继承 band importance 或 mean spectrum

因此聚合图更适合回答：

- 在整一层上，哪些波段最稳定地重要
- 不同 `parent` 子模型的关注模式是否一致
- 某些类别的解释结果是否只在个别子模型里成立，还是具有跨 parent 的稳定性

### 8.6 独立测试集 embedding 诊断

除了 `analyze.py` 这条训练后解释主线，仓库根目录还提供了：

- `compare_test_train_means.py`

这个脚本不是训练期解释工具，而是独立测试集诊断脚本

围绕 embedding 做三类对照：

- 测试谱对训练谱的最近邻诊断
- 测试文件夹 centroid 对训练类 centroid 的相似度诊断
- 文件夹级模型投票结果汇总

脚本会读取：

- 实验目录中的模型权重
- `dataset_train/` 作为训练 embedding bank
- `dataset_test/` 作为独立测试样本来源

最终输出目录通常为：

```text
<EXP_DIR>/test_train_embedding_compare/
```

其中通常包含：

- `summary.csv`
- 每个测试文件夹一张柱状图

每张图默认有两列柱状图：

- 左图 `Model Vote Top-K`
  - 每条测试谱先走模型分类头
  - 取各条谱的 `top1`
  - 再在文件夹级别做投票统计
- 右图 `Embedding Neighbor Vote Top-K`
  - 每条测试谱先提取 embedding
  - 再到训练 embedding bank 中找最近邻
  - 用最近训练样本的类别做投票统计

图题格式通常为：

```text
<folder_name> | expected=<expected_label> | model_top1=<...> | neighbor_top1=<...> | centroid_top1=<...>
```

其中：

- `expected` 是根据测试文件夹名推断出的理论类别
- `model_top1` 是模型多数票第一名
- `neighbor_top1` 是 embedding 最近邻多数票第一名
- `centroid_top1` 是测试文件夹平均 embedding 与训练类 centroid 做余弦相似度后的第一名

这三者不一定一致，而这种不一致本身就是诊断信息。例如：

- `model_top1` 错，但 `neighbor_top1` 对，更像分类头没有把已有 embedding 信息充分用好
- `model_top1` 和 `neighbor_top1` 都错到同一类，更像测试样本在特征空间里本来就更贴近错误类
- `neighbor_top1` 对，但 `centroid_top1` 错，更像文件夹内部是多峰结构，简单取平均抹平了细节
- `centroid_top1` 对，但 `model_top1` 错，更像整体中心仍然正确，但单谱层面边界样本较多

因此，这个脚本的价值不在于“再看一次预测结果”，而在于帮助判断问题更可能来自：

- 分类头本身
- embedding 空间本身
- 测试分布相对训练分布的偏移

### 8.7 这些分析结果如何解读

对拉曼光谱任务来说，这些分析结果常常比单个 Accuracy 更有解释价值，因为它们在回答下面这些更具体的问题：

- 模型到底依赖了哪些波段和哪些输入通道
- 不同类别的判别主要依赖局部峰位，还是依赖多个峰之间的组合关系
- embedding 空间里类别是否真的被分开，还是只是分类头勉强分开
- 多层级拆模后，不同 `parent` 子模型的关注区域是否发生迁移
- 模型学到的是稳定的谱学特征，还是偶然噪声和数据偏差

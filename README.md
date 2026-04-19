# 拉曼光谱层级分类项目

## 1. 项目目标

本项目面向细菌拉曼光谱识别任务，构建一套完整的层级分类实验系统

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

因此，这个项目的真实任务可以概括为：

1. 把原始拉曼光谱统一到可训练的标准表示空间
2. 在该表示空间上学习稳定的层级判别边界
3. 在预测时按层级逐级细化
4. 用分析工具判断模型到底在看哪些峰段、哪些层、哪些通道

项目当前的技术主线如下：

1. 离线预处理阶段：做 AsLS 基线校正、波段裁剪、坏段剔除、统一波数轴插值、训练集 PCA 异常值过滤
2. 在线输入阶段：从清洗后的单通道光谱构造模型输入
3. 模型阶段：使用多尺度 1D CNN 主干提取局部峰形，再接序列编码器和分类头
4. 训练阶段：围绕层级标签、类别不均衡和细粒度难样本设计多种损失与重加权策略
5. 评估与分析阶段：通过混淆矩阵、embedding 近邻诊断、IG、Layer Grad-CAM 等方式分析错误来源

## 2. 仓库结构与模块职责

```text
拉曼光谱分类/
├─ train.py                          # 顶层训练入口，只负责当前训练层级和手动覆盖项
├─ evaluate.py                       # 测试集评估入口
├─ pca_svm_baseline.py               # PCA + SVM 基线入口
├─ analyze.py                        # 统一分析入口（single / aggregate）
├─ Independent_test.py               # 独立测试集 embedding 近邻诊断入口
├─ raman/
│  ├─ config.py                      # 训练配置定义：输入、模型、损失、增强、优化器参数
│  ├─ config_io.py                   # config.yaml 读写与实验配置回载
│  ├─ model.py                       # 主模型实现：多尺度 stem + 1D CNN + encoder + pooling + head
│  ├─ trainer.py                     # 训练主流程：建数据集、建模型、训练、验证、保存结果
│  ├─ data/
│  │  ├─ paths.py                    # 训练/测试目录解析，统一把 dataset_root 映射到具体阶段目录
│  │  ├─ dataset.py                  # 层级数据集扫描、标签编码、样本索引与 DataLoader 输入接口
│  │  └─ preprocess.py               # 在线预处理与增强：标准化、smooth/d1 通道构建、训练增强
│  ├─ analysis/
│  │  ├─ pipeline.py                 # 分析主调度：构建上下文、组织 single / aggregate 任务与输出
│  │  ├─ ig.py                       # Integrated Gradients：输入通道重要性与类别波段重要性
│  │  ├─ gradcam.py                  # Layer Grad-CAM：层级/分组重要性分析
│  │  ├─ embedding.py                # embedding 收集与 train + test 联合可视化
│  │  └─ se.py                       # 读取训练期 SE sidecar 并输出 SEBlock 缩放统计
│  ├─ eval/
│  │  ├─ experiment.py               # 实验目录、配置、hierarchy meta 与模型路径解析
│  │  ├─ runtime.py                  # 实验运行时：模型懒加载、缓存与 SE sidecar 读取
│  │  ├─ common.py                   # 共享推理辅助：logits 选择、层级掩码、级联推理、指标计算
│  │  ├─ evaluator.py                # 测试集评估主流程与结果落盘
│  │  ├─ baseline.py                 # PCA + SVM 基线实现
│  │  └─ report.py                   # classification report、混淆矩阵、文本结果输出
│  └─ training/
│     ├─ split.py                    # 训练/验证切分、训练范围解析、父类过滤
│     ├─ losses.py                   # Focal / SupCon / AlignLoss / class weight 等损失工具
│     └─ session.py                  # 训练会话初始化：随机种子、输出目录、日志、配置快照
├─ dataset_process/
│  ├─ cli.py                         # 离线数据处理统一入口，负责 pack/classify/preprocess/count
│  ├─ profiles.py                    # 各数据集的目录布局、原始目录命名、统一坏波段设置
│  ├─ common.py                      # 离线预处理底层函数：读谱、AsLS、坏段掩码、单谱清洗、均值谱绘图
│  └─ pipeline.py                    # 离线主流程：打包、目录重组、训练集清洗、测试集清洗、统计
├─ predict/
│  ├─ predict_core.py                # 层级级联推理核心
│  ├─ predict_folder.py              # 批量目录预测
│  └─ predict_single.py              # 单目录预测
├─ colab/
│  └─ colab_unified.ipynb            # Colab 一体化 notebook：解压库、数据处理、训练、评估、分析、打包
├─ notebooks/
│  └─ single_process_AsLS_cut_SNV.ipynb # 单条光谱从原始输入到模型通道构建的可视化 notebook
└─ dataset/
   ├─ 细菌/
   ├─ 耐药菌/
   └─ 厌氧菌/
```

## 3. 记号说明

- $n$：样本索引，常用于表示第 $n$ 个样本
- $i$：位置索引，表示光谱序列中的第 $i$ 个波数采样点
- $c$：类别索引，表示第 $c$ 类
- $k$：PCA主成分个数
- $K$：类别总数
- $t$：训练轮次（epoch）或时间步索引
- $x$：输入光谱、输入特征或中间表示
- $y$：真实标签

## 4. 离线数据预处理

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

### 4.1 参数修改

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

### 4.2 数据集目录结构

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

### 4.3 原始数据预览

- 直接基于 `dataset_init/` 或 `dataset_init.npz` 做预处理预览
- 执行基线校正、裁剪、坏波段剔除与统一参考轴插值
- 不做 PCA 异常值过滤与光谱输出
- 只输出每个分组的均值谱图到 `dataset_init_fig/`

先检查原始数据质量，看是否需要将某个文件夹移除，不再进入后续训练数据集

### 4.4 重组数据集

由于采集数据按日期划分，原始数据集一般命名为`类别+数字`

所以需要重组数据，把多个文件夹按类别收缩到一个文件夹中

- 扫描 `dataset_init/` 或 `dataset_init.npz`

- 读取叶子目录名

- 统一按 `letters_sign` 规则提取类别前缀
  
  例如：`ABC12 -> ABC`，`ESBL+03 -> ESBL+`
  
- 将样本复制到 `dataset_train_raw/`，文件名更改为`叶子目录名_原文件名`

这一步的目的是先把原始采集目录整理成更稳定的类别目录结构，供后续统一清洗

### 4.5 训练集离线清洗

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

设原始光谱为 $x \in \mathbb{R}^L$，基线为 $b \in \mathbb{R}^L$，则 AsLS 的优化目标可写为

```math
\min_{b} \sum_{i=1}^{L} w_i (x_i - b_i)^2 + \lambda \sum_{i=1}^{L-2} (b_{i+2} - 2b_{i+1} + b_i)^2
```

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

- $\lambda$为平滑参数，控制基线光滑程度

在权重固定时，对目标函数关于 $b$ 求导并令其为零，可得到线性方程组

```math
(W + \lambda D^T D) b = W x
```

```python
matrix_b = (matrix_w + lam * (D.T @ D)).tocsc()     # W + λD^T D
baseline = spsolve(matrix_b, weights * spectrum)    # 解 b
```

由于权重 $w_i$ 本身依赖于当前基线估计 $z$，因此 AsLS 需要通过迭代方式求解

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
T_{k} = \bar X P_{k}
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

离线阶段完成以后，`dataset_train/` 和 `dataset_test/` 中保存的是“已经完成基线校正、坏段剔除、统一波数轴对齐”的单条光谱文本文件

这些单条光谱进一步转换成模型真正使用的输入张量:

1. 从 `dataset_train/` 扫描目录树
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

### 5.3 模型输入

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

### 5.4 训练集、验证集、测试集

- 训练入口使用的基础数据目录是 `dataset_train/`
- 训练集与验证集都从 `dataset_train/` 内部划分得到
- 如果实验目录下已有 `train_files.json` 和 `test_files.json`，会优先复用原切分
- 如果没有，就按 `split_level` 重新分组切分

当前训练代码中：

- `train_dataset = RamanDataset(..., augment=True)`，用于训练
- `test_dataset = RamanDataset(..., augment=False)`，用于训练过程中的验证

这里的 `test_dataset` 只是训练阶段的验证集视图，并不等同于外部独立测试集

真正的独立测试集位于 `dataset_test/`，不参与训练期切分

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
    # bottleneck 主支路 + shortcut 残差支路
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
scale = self.pool(x).view(batch_size, channels)
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
$$
\widetilde{x}_{b,c,i} = \mathrm{scale}_{b,c}  \odot x_{b,c,i}
$$
```python
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
PE(pos, 2i) = \sin \left(pos \cdot 10000^{-2i / d_{\text{model}}}\right)\\
PE(pos, 2i+1) = \cos \left(pos \cdot 10000^{-2i / d_{\text{model}}}\right)
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

- 数据集完整层级始终由 `dataset_train/` 的目录树自动扫描得到
- `CURRENT_TRAIN_LEVEL` 只表示“这次训练实际要训练的那一层”

当 `train_per_parent=True` 时，训练行为是：

- 顶层没有父层，因此训练全局模型
- 若当前层存在父层，则按父类拆成多个子任务
- 若某个父类下只有一个子类，则不训练该 parent 子模型，只在层级元数据中记录这条确定关系

如果当前实验目录缺少上一级模型或单子类记录，训练开始时会打印提示，提醒先训练哪一级

若给定 `TRAIN_ONLY_PARENT`，则直接按父类索引限制训练范围

### 7.3 训练/验证切分

训练代码扫描的是 `dataset_train/`，然后在内部再做 train/val 切分

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

其中

- $\gamma$：控制对易样本的抑制程度

```python
criterion = FocalLoss(
    gamma=config.gamma,
    weight=base_class_weights,
    ignore_index=-1,
)
```

`FocalLoss.forward()` 的实现是先基于未加权交叉熵计算 $p_t$ 和 focal 因子，再用当前 epoch 的类别权重对逐样本损失做重加权

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

`FocalLoss` 返回的是逐样本 loss 向量，后续才能继续叠加 `severity weight`

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
       mask = (y_valid == c)
       if mask.any():
           mean_ce = ce_each[mask].mean()
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
  - 真实类排第四及以后，样本权重为 `1.10`

其中高置信度阈值也按类别数调整：

- 三分类使用 `0.88`
- 四类及以上使用 `0.85`

与 `FocalLoss` 形成互补：`FocalLoss` 更关注概率难度，`severity weight` 更关注错误排序结构

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

对应的类内紧凑项为

```math
L_c^{(\text{align})}=\frac{1}{|S_c|}\sum_{n \in S_c}\|x_c - \mathrm{center}_c^{(\text{batch})}\|_2^2
```

对当前 batch 内所有有效类别取平均

```math
L_{\text{align}}
=
\frac{1}{|C_{\text{valid}}|}
\sum_{c \in C_{\text{valid}}}
L_c^{(\text{align})}
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

2. 计算温度缩放后的两两相似度

   ```math
   \mathrm{sim}(n,m) = \frac{z_n^\top z_m}{\tau}
   ```

3. 定义正样本集合：

   ```math
   P(n)=\{\,m \mid m\neq n,\; y_m = y_n\,\}
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
- `SupConLoss` 解决类间分离与类内相对距离结构
- `AlignLoss` 解决当前 batch 内的类内紧凑性

当前训练闭环可以概括为：

- 用 `FocalLoss` 学分类边界
- 用 `base_class_weights` 做静态类别平衡
- 用 `ema_class_weights` 在训练中后期按类别难度动态修正权重
- 用 `severity weight` 提高高置信错判样本的学习强度
- 用 `SupConLoss` 拉开 embedding 的相对结构
- 用 `AlignLoss` 收紧当前层的类内分布

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

1. 从 `dataset_train/` 中按训练时的 split 提取 train/test 样本
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

### 8.6 独立测试集分析

根目录下`Independent_test.py` 面向 `dataset_test/` 的独立测试集做评估

从文件夹级别判断外部测试样本与训练分布之间的接近程度

对每个测试文件夹，从“模型预测”、“embedding 最近邻投票”、“类别质心相似度”三个角度交叉分析当前测试文件夹更像哪一类

在分析测试文件夹之前，脚本会先用训练集建立一个对照用的 embedding bank

1. 读取当前实验目录对应的训练切分 `train_indices`

2. 用训练好的模型提取训练样本在 `COMPARE_LEVEL` 下的特征

3. 对特征做 L2 归一化

   ```python
   _, feat = model(xs, return_feat=True)
   feat = _l2_normalize_rows(feat).cpu()
   ```

4. 保存训练 embedding 及其类别标签

   ```python
   center = train_feats[mask].mean(dim=0, keepdim=True)
   centroids[class_id] = _l2_normalize_rows(center)[0]
   ```

   把训练集中每个类别在特征空间中的“代表方向”提取出来

**模型投票（Model Vote）**

统计该文件夹中每条光谱被模型预测成各类别的次数，并形成投票分布

反映的是：模型直接把这个文件夹中的样本整体看成哪一类

**embedding 最近邻投票**

看测试样本在 embedding 空间中最接近哪些训练样本

先计算测试 embedding 与训练 embedding bank 的相似度矩阵

```python
similarity = torch.matmul(folder_feats, train_feats_t)
nearest_indices = torch.argmax(similarity, dim=1).cpu().numpy()
neighbor_preds = train_labels[nearest_indices].astype(np.int64)
```

由于训练 embedding 已经做过 L2 归一化，这里的点积就是余弦相似度

对每个测试样本找到相似度最高的训练样本，把该训练样本的类别当作最近邻类别

反映的是：只看特征空间中的最近邻结构，这个测试文件夹更像训练集中的哪一类

**类别质心相似度**

对一个测试文件夹内部所有样本的 embedding 求均值，再做 L2 归一化，得到该文件夹的平均特征中心

再去与各类别质心计算余弦相似度

反映的是：从整个文件夹的平均特征来看，它整体更接近训练集中的哪个类别原型

输出目录为：

```text
<EXP_DIR>/embedding_compare/
```

其中每个测试文件夹会单独保存：

- `spectra.png`：测试谱形与期望训练均值谱的对照图
- `model_vote.png`：模型预测投票分布
- `neighbor_vote.png`：embedding 最近邻投票分布
- `centroid_similarity.png`：文件夹平均 embedding 与各类别质心的相似度条形图

可能的结果：

1. 模型投票错，但最近邻和质心接近期望类

   说明分类头可能更容易受边界细节影响，而 embedding 本身仍较接近期望类

2. 模型投票、最近邻投票和质心相似度都偏向同一个错误类

   该文件夹整体在特征空间里已经更像另一个类别，可能是分布偏移或标签问题

3. 均值谱明显偏离期望类训练均值谱

   不仅是分类边界问题，还可能存在数据采集条件差异、预处理不一致或样本本身差异

## 9. 分析

### 9.1 模式

分析入口保留在仓库根目录`analyze.py`

支持两种模式：

- `single`：对一个具体模型做完整分析
  
  如果当前层没有全局模型，也可以自动退化为对多个 `parent` 子模型分别分析
- `aggregate`：面向“该层按 parent 拆模训练”的场景
  
  逐个加载所有 `parent` 子模型，分别计算解释结果，再做加权聚合

### 9.2 输出

分析会围绕一个具体模型输出多种解释结果，核心包括：

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

# Laplacian-Based Focus Stacking Plan

本方案先实现一个基于局部 Laplacian 清晰度选择的基础焦点堆叠算法，统一处理指定刀具或指定视角，保存融合图、评价 CSV 和评分柱状图。

> 本文记录已确定的实施方案，不表示相关代码已经实现。配准的具体估计方法、图像读取格式及评价区域的指定方式，在实现时进一步明确。

## 1. 输入与运行方式

原始图像按以下结构组织：

```text
data/raw/
└── 具体刀具名称/
    ├── flank/
    ├── rake/
    └── face/
```

当前按一个视角目录包含一个焦点序列处理，不混合不同视角。若后续一个视角包含多个独立序列，应明确区分序列，不直接混合融合。

计划支持两种运行方式：

```powershell
# 依次处理该刀具的 flank、rake、face
python run_benchmark.py --input "data\raw\Kro_DEMO_10" --method all

# 只处理 face
python run_benchmark.py --input "data\raw\Kro_DEMO_10\face" --method laplacian
```

更换刀具时，手动更换输入路径中的刀具名称。

方法选择支持 `laplacian`、`method_a`、`method_b` 和 `all`，默认 `all`。Method A、Method B 继续保留占位，不自行确定算法。

## 2. 统一处理流程

由 `run_benchmark.py` 组织：

1. 读取并固定焦点序列的图像顺序。
2. 将彩色图像配准到中间帧的尺度，必要时补偿少量平移。
3. 确定公共有效区域，排除配准产生的无效边界。
4. 从配准后的彩色图像生成灰度副本。
5. 调用选定算法，获得彩色融合图。
6. 对融合图计算两个清晰度指标。
7. 保存结果图、CSV 和柱状图。

即使刀具没有发生物理横向移动，尺度变化的中心也可能不与刀具中心重合，因此配准仍可能需要少量平移补偿。

配准和灰度转换作为共用功能，第一版在 `run_benchmark.py` 中定义辅助函数：

```python
align_images(images)
to_gray_images(images)
```

所有算法使用相同的输入选择、配准结果和评价流程。

## 3. Laplacian 算法

`methods/laplacian.py` 内部负责四项工作：

| 工作 | 初版方案 |
| --- | --- |
| 局部清晰度计算 | 灰度图的 Laplacian 绝对值，在 21×21 窗口内求均值 |
| 来源选择 | 每个位置选择清晰度分数最高的图像编号 |
| 空间平滑 | 对来源编号图进行 5×5 多数投票 |
| 彩色融合 | 按平滑后的编号图，提取对应彩色图像中的完整 BGR 像素 |

### 3.1 局部清晰度计算

设第 $i$ 张配准后的灰度图为 $G_i$，以位置 $(x,y)$ 为中心的 21×21 窗口为 $W_{21}(x,y)$。局部清晰度为：

$$
S_i(x,y)=\frac{1}{441}
\sum_{(u,v)\in W_{21}(x,y)}
\left|\nabla^2 G_i(u,v)\right|
$$

每张输入图得到一张与输入尺寸相同的局部清晰度图。

计算 Laplacian 的初始核大小采用 3；它与汇总清晰度的 21×21 窗口是两个不同参数。Laplacian 响应使用浮点类型，保留正负值后再取绝对值。

### 3.2 来源选择

对各张图在同一位置的清晰度分数进行比较：

$$
M(x,y)=\operatorname*{arg\,max}_{i} S_i(x,y)
$$

$M(x,y)$ 表示位置 $(x,y)$ 的来源图像编号。代码中的编号从 0 开始；例如编号 4 对应第 5 张输入图。分数完全相同时，选择输入顺序中靠前的图像，因此输入顺序必须固定。

### 3.3 空间平滑

初版采用 5×5 多数投票：统计邻域内各来源编号出现的次数，将中心编号替换为出现次数最多的编号。

应从原始来源编号图计算新的编号图，避免边修改边影响后续位置。编号是离散类别，不直接对编号求平均。

平滑处理的是来源选择，不是彩色图像中的纹理。窗口过大可能吞掉细小的真实清晰区域。

### 3.4 彩色融合与接口

按照平滑后的来源编号图，从对应彩色图像中提取完整的 BGR 像素。第一版直接选择像素，不加入边界加权融合。

清晰度窗口 21×21 与平滑窗口 5×5 保持为可调整参数；后续可比较清晰度窗口 11×11、21×21、31×31，以及不同平滑设置。

统一调用形式：

```python
fused = laplacian.stack(
    aligned_images,
    gray_images=gray_images,
    window_size=21,
    smooth_size=5,
)
```

`stack()` 返回融合彩图，来源编号图是内部中间结果。算法不得修改输入的彩色图或灰度副本。

## 4. 清晰度评价

在现有 `evaluations/sharpness.py` 中实现：

```python
tenengrad(image, **kwargs)
laplacian_variance(image, **kwargs)
evaluate(image, **kwargs)
```

初版只评价融合图，分别输出两个数值，不合成总分。暂时不评价输入图，也不计算融合前后的变化率。

设融合图的灰度图为 $I$，评价区域为 $\Omega$，像素数量为 $N=|\Omega|$。

### 4.1 Tenengrad

Tenengrad 作为主要清晰度指标，采用无阈值的平均 Sobel 梯度能量。

先计算水平和垂直梯度：

$$
G_x=K_x*I,\qquad G_y=K_y*I
$$

其中 $*$ 表示滤波运算，3×3 Sobel 核为：

$$
K_x=
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix},
\qquad
K_y=
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$

最终评价值为：

$$
T=\frac{1}{N}
\sum_{(x,y)\in\Omega}
\left[G_x(x,y)^2+G_y(x,y)^2\right]
$$

它表示评价区域内的平均梯度能量。这里采用均值版本，避免分数直接随像素数量增加。

### 4.2 Laplacian 方差

Laplacian 方差作为辅助清晰度指标。

先计算有符号的 Laplacian 响应及其在评价区域内的均值：

$$
L(x,y)=\nabla^2 I(x,y)
$$

$$
\bar L=\frac{1}{N}
\sum_{(x,y)\in\Omega} L(x,y)
$$

最终评价值为：

$$
V_L=\frac{1}{N}
\sum_{(x,y)\in\Omega}
\left[L(x,y)-\bar L\right]^2
$$

这里不先对 Laplacian 响应取绝对值。它与融合算法内部的“局部 Laplacian 绝对值均值”是不同的计算。

### 4.3 评价约定

- 所有方法使用相同评价区域、灰度范围、滤波参数和边界处理。
- 评价区域优先选择刀具前刀面，并排除无效边界。
- 若输入为 8 位图像，可统一除以 255 转为浮点灰度；不对各图分别做最大最小值归一化。
- 两个指标分别保存，不直接相加，不转换为未经定义的百分制评分。
- 噪声、接缝也可能提高分数，因此不能只凭清晰度数值判定融合质量。
- 没有全清晰真值时，不将普通 SSIM 或 PSNR 作为主要绝对评价指标。

`evaluate()` 返回：

```python
{
    "tenengrad": T,
    "laplacian_variance": V_L,
}
```

## 5. 人工评价、耗时与失败

### 5.1 人工评价

接缝、伪影和细节保留通过固定局部区域人工检查，记录重影、斑点、颜色跳变、细节丢失及对应位置。

初版不实现自动伪影评分，`evaluations/artifacts.py` 保留明确占位，不返回虚假的评价值。人工评价记录保存在 `results/`，并与对应方法和融合图关联。

### 5.2 融合耗时

融合耗时由 `run_benchmark.py` 使用 `perf_counter()` 测量，定义为：

> 从调用 `stack()` 开始，到返回融合图结束。

包含算法内部的清晰度计算、来源选择、平滑和彩色融合；排除读取、配准、共用灰度转换、评价和保存。

```python
from time import perf_counter

start = perf_counter()

fused = laplacian.stack(
    aligned_images,
    gray_images=gray_images,
    window_size=21,
    smooth_size=5,
)

fusion_seconds = perf_counter() - start
```

所有方法采用相同的计时范围。

### 5.3 失败与状态

- 无效输入或运行异常，记录失败阶段及原因。
- 指定刀具目录时，缺失或为空的视角记录原因，继续处理其他视角。
- 指定单个视角时，路径无效或没有可用图像，明确报告错误。
- 算法发现无效输入时抛出明确异常，由统一入口捕获常规异常并记录。
- 正常返回图像但质量较差，属于执行成功，由人工评价记录质量问题。
- 尚未实现的方法明确标记，不生成虚假图像或评分。
- 不通过随意返回某张输入图来掩盖失败。

## 6. 输出目录与文件

每个视角单独保存到：

```text
results\具体刀具名称\视角\
```

例如：

```text
results/
└── Kro_DEMO_10/
    ├── flank/
    │   ├── laplacian.png
    │   ├── method_a.png
    │   ├── method_b.png
    │   ├── benchmark.csv
    │   └── metrics_comparison.png
    ├── rake/
    │   └── ...
    └── face/
        └── ...
```

各算法结果图仅在成功运行后生成。当前 Method A、Method B 尚未实现，因此不实际生成对应结果图。

### 6.1 CSV

每个视角一个 `benchmark.csv`，每次运行中，每个被选择的算法对应一行，字段为：

| 字段 | 内容 |
| --- | --- |
| `tool` | 刀具名称 |
| `view` | 视角 |
| `method` | 算法名称 |
| `image_count` | 输入图像数量 |
| `tenengrad` | Tenengrad 评分 |
| `laplacian_variance` | Laplacian 方差 |
| `fusion_seconds` | 融合耗时，单位秒 |
| `status` | 成功、失败或尚未实现 |
| `error` | 失败原因，成功时留空 |

不包含 `input_files`、`parameters`、`focus_step`。失败或尚未实现的方法，评分留空，不填成 0。

### 6.2 评分柱状图

每个视角生成一个 `metrics_comparison.png`，包含两个并排子图：

- 左图：各算法的 Tenengrad。
- 右图：各算法的 Laplacian 方差。

两个指标分别使用各自的数值尺度，不合成总分。

即使只有一个成功算法，也生成图，每个子图显示一根柱。失败或尚未实现的方法注明状态，不绘制零分柱；全部没有有效结果时，生成注明“无有效评分结果”的图。

## 7. 文件职责与范围

| 文件或目录 | 职责 |
| --- | --- |
| `run_benchmark.py` | 唯一执行入口，共用处理、调用、计时、保存和绘图 |
| `methods/laplacian.py` | 局部清晰度、来源选择、平滑和彩色融合 |
| `methods/method_a.py`、`methods/method_b.py` | 候选方法占位 |
| `evaluations/sharpness.py` | 两个清晰度指标及统一评价接口 |
| `evaluations/artifacts.py` | 自动伪影评价占位 |
| `data/raw/` | 只读原始输入 |
| `results/` | 所有生成结果及人工评价记录 |

保持现有简单结构，不增加独立运行脚本、GUI、数据库或采集控制代码，不使用显微镜数据或显微镜真值。

原始数据不修改，生成结果只写入 `results/`。正式 Git 交付时，明确选择需要纳入版本管理的结果文件，避免因忽略规则遗漏交付内容。

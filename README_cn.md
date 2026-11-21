# SG3 Focus Stacking

## 项目简介

本项目将同一刀具视角下、不同焦平面的照片融合为景深更大的图像，并比较四种焦点堆叠算法。
统一入口负责读取、配准、融合、评价与结果保存，各方法使用相同输入和评价区域。

| 算法 | 命令行名称 | 依据 |
|---|---|---|
| [局部 Laplacian（Baseline）](methods/laplacian.py) | `laplacian` | 局部清晰度选择与多数投票平滑 |
| [GFDF 引导滤波融合](methods/guided_filter.py) | `guided_filter` | [Qiu 等，2019](https://doi.org/10.1016/j.image.2018.12.004)；[作者代码](https://github.com/bitname/Multi-focus-image-fusion-GFDF) `40c2f25` |
| [Laplacian 金字塔融合](methods/laplacian_pyramid.py) | `laplacian_pyramid` | [Wang、Chang，2011](https://www.jcomputers.us/vol6/jcp0612-07.pdf) |
| [IFCNN-MAX](methods/ifcnn.py) | `ifcnn` | [Zhang 等，2020](https://doi.org/10.1016/j.inffus.2019.07.011)；[作者代码](https://github.com/uzeful/IFCNN) `7dd4a2a` |

- **局部 Laplacian**：根据局部清晰度选择源图像像素，经多数投票平滑选择结果后合成图像。
- **GFDF 引导滤波融合**：检测各帧的清晰区域，再通过引导滤波细化融合权重，平滑连接不同帧的清晰内容。
- **Laplacian 金字塔融合**：将图像分解为多个尺度，在各层选择信息更丰富的内容，再逐层重建融合图像。
- **IFCNN-MAX**：使用预训练卷积网络提取各帧特征，逐元素取最大值融合特征后重建图像。

## 安装

需要 Python 3.10 或更新版本。以下命令在项目根目录的 PowerShell 中执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

基础依赖不包含 PyTorch，只有IFCNN需安装它。

**IFCNN 额外要求：** NVIDIA CUDA GPU、支持 CUDA 的 `torch>=2.4` 和作者权重。下面以 CUDA 12.6 为例，按本机驱动选择兼容构建，参见 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)：

```powershell
.venv\Scripts\python.exe -m pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```


下载 [IFCNN-MAX.pth](https://github.com/uzeful/IFCNN/raw/1d281720468bb0c317957b6bd26929ae45a9dfb5/Code/snapshots/IFCNN-MAX.pth)，保存到 `methods/weights/IFCNN-MAX.pth`。加载时校验 SHA-256：

```text
6022b44c88a2f0367ba9b08fcdf0d663745d9a966fca0ee1f41867797b6ff53f
```

## 运行

将 `tool_01` 替换为实际刀具目录名，先按下文的数据结构放置图像：

```powershell
# 运行一个刀具一个视角的 Baseline
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01\view1" --method laplacian

# 分别运行以下刀具所有视角的全部四种算法（自动发现各刀具目录下的 view<N>）
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Alu_WSP_12x80" --method all --roi 600 200 1100 1300
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Garant_202244_D10" --method all --roi 600 200 1100 1100
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\GoMill-D6-L100" --method all --roi 600 200 1100 1200
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Kro_DEMO_10" --method all --roi 600 200 1100 1200
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Kro_FM16_8mm_6" --method all --roi 600 200 900 1200
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Krue_PIN_MAPAL_d12" --method all --roi 600 200 1100 1200
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Krue_Stahl_10mm_Plan" --method all --roi 600 200 1100 1200

# 查看全部命令参数
.venv\Scripts\python.exe run_benchmark.py --help
```

`--method` 可选上表中的四个名称或 `all`，默认 `all`。`--input` 必填，可同时给出多个刀具或视角目录，重复视角只处理一次。
`--roi X Y WIDTH HEIGHT` 可设置用于评分的矩形区域

候选算法默认参数集中在各算法文件顶部的 `DEFAULT_PARAMETERS`，显式覆盖优先；实际参数和按尺寸调整后的设置写入 `run_info.json`。扫描结果保存在对应视角的 `sweep_<参数名>/` 子目录。

## 数据结构

输入来自外部已采集的刀具焦点序列；每个视角目录存放一个独立序列：

```text
data/raw/
└── tool_01/
    ├── view1/
    │   ├── frame_1.png
    │   └── frame_2.png
    └── view2/
        ├── frame_1.png
        └── frame_2.png
```

- 视角名为小写 `view` 加无前导零的正整数，如 `view1`、`view3`、`view10`；按数字排序。
- 支持 PNG、JPEG、TIFF、BMP 三通道 `uint8` / `uint16` 彩色图像；同一序列尺寸和位深须一致，文件名按自然顺序排序。

## 输出与评价

结果保存到 `results/<tool>/view<N>/`：

| 文件 | 内容 |
|---|---|
| `<方法名>.png` | 成功方法的融合图 |
| `benchmark.csv` | 输入数量、三项评价指标、融合耗时、状态及错误 |
| `run_info.json` | 完整输入顺序、实际参数（含梯度过冲参数）、配准和环境信息 |
| `metrics_comparison.png` | 三项评价指标对比图，每项一个子图 |
| `roi_preview.png` | 指定 ROI 时生成的参考图 |

### 评价指标

三项指标都在归一化到 [0, 1] 的灰度图上计算。默认先在整幅图上计算，再取有效区域的值；指定 `--roi` 时只取 ROI 内的值。有效区域都排除外围 1 像素。

| 指标 | 文件 | 含义 | 方向 |
|---|---|---|---|
| `tenengrad` | [sharpness.py](evaluations/sharpness.py) | 3×3 Sobel 梯度平方和的平均值，强边缘占比更大 | 越高越清晰 |
| `average_gradient` | [sharpness.py](evaluations/sharpness.py) | 平均梯度 AG | 越高越清晰 |
| `gradient_overshoot_ratio` | [artifacts.py](evaluations/artifacts.py) | 融合图边缘强于所有源帧的像素比例 | 越低越好 |

**平均梯度 AG**：依据 [Kumar，SIViP 2013](https://doi.org/10.1007/s11760-012-0361-x) 第 3.3 节第 3 条：

$$AG=\frac{1}{mn}\sum_i\sum_j\sqrt{\big(f(i,j)-f(i+1,j)\big)^2+\big(f(i,j)-f(i,j+1)\big)^2}$$

采用前向差分，根号内不除以 2，与原文一致；其他文献中常见的"除以 2"变体，数值是本式的 1/√2。原文使用 8 位灰度，本项目使用 [0, 1] 灰度，数值为原文尺度的 1/255。原文未说明最后一行和最后一列的处理方式，本项目在与 Tenengrad 相同的有效区域内求平均，该区域内每个像素都有前向邻居，这一点属于工程处理。AG 与 Tenengrad 都基于梯度，两者高度相关：Tenengrad 对梯度取平方，强边缘权重更大；AG 是梯度幅值的平均值，便于和文献比较。

**梯度过冲比例**：依据同一论文式 (21) 中的伪影指示函数 AM（源自 Petrović 和 Xydeas，ICCV 2005）：如果融合图某处的边缘强度大于所有源图像在该处的边缘强度，该处就记为伪影。这类梯度不来自任何源帧，通常对应拼接接缝、光晕、振铃或被放大的噪声。本项目只实现了指示函数部分，没有计算完整 N_m^{AB/F} 中的 (1 − Q) 和感知权重。判定规则如下：

- 边缘强度 G 为 3×3 Sobel 梯度幅值，√(Sx² + Sy²)。
- G_max 为同一位置所有配准后源帧边缘强度的最大值，再做最大值膨胀。
- 满足 G_融合 > (1 + k)·G_max + ε 的像素记为过冲，统计这类像素在评价区域内所占的比例。

从两张源图扩展到 N 帧、容差 k、下限 ε 和膨胀窗口都属于工程适配，参数位于 `artifacts.py` 顶部的 `DEFAULT_PARAMETERS`：

| 参数 | 默认值 | 含义 | 调大的影响 / 调小的影响 |
|---|---|---|---|
| `overshoot_factor`（k） | 0.1 | 相对余量：比最清晰的源帧陡 10% 以上才计为过冲 | 忽略更多轻微锐化 / 插值、取整差异也会被计入 |
| `overshoot_floor`（ε） | 0.02 | 绝对下限，单位为 [0, 1] 灰度上的 Sobel 幅值（全黑到全白的阶跃为 4），约相当于 1 到 2 个 8 位灰度级 | 忽略平坦区域的弱过冲 / 平坦区域的噪声也会被计入 |
| `source_max_window` | 3 | 源帧最大边缘强度的膨胀窗口（像素，正奇数，1 表示不膨胀），用于容忍约 1 像素的配准残差 | 能容忍更大的错位，但可能掩盖强边缘旁的接缝 / 判定更严格 |

以上初始值都是工程设定，不是文献值，也不是经过验证的最优值。

**解读原则**：

- 清晰度指标回答"细节有多少"，过冲比例回答"细节是不是真的"，三项指标不合成总分。如果清晰度较高、同时过冲也较高，说明分数可能被伪影抬高，需要结合局部目视检查判断。Kumar 在论文第 4 节也指出，AG 等指标最高的融合图，目视质量可能反而更差。
- 金属高光在不同焦平面下会变化，配准残差也会形成真实的重影，两者都可能被计为过冲。结论应当结合 ROI 目视检查做出。

新运行的 CSV 和对比图不再输出 Laplacian 方差，历史结果保持原样。融合耗时不含公共处理、评价和保存，IFCNN 模型加载与预热另记；GPU 与 CPU 耗时不能直接比较快慢。


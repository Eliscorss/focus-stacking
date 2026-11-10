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
# 运行一个视角的 Baseline
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01\view1" --method laplacian

# 运行该刀具所有视角的全部四种算法
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01" --method all

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
| `benchmark.csv` | 输入数量、清晰度指标、融合耗时、状态及错误 |
| `run_info.json` | 完整输入顺序、实际参数、配准和环境信息 |
| `metrics_comparison.png` | 清晰度指标对比图 |
| `roi_preview.png` | 指定 ROI 时生成的参考图 |

评价采用 Tenengrad 和 Laplacian 方差，需结合局部目视检查；自动接缝与伪影评价尚未实现。融合耗时不含公共处理、评价和保存，IFCNN 模型加载与预热另记；GPU 与 CPU 耗时不能直接比较快慢。


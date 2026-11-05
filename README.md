# SG3 Focus Stacking

本项目用于 SG3 刀具图像的焦点堆叠与算法比较。将同一视角下、不同焦平面的多张照片融合为一张景深更大的图像，以保留弯曲刀具前刀面不同位置的清晰细节。

程序通过统一入口完成图像读取、配准、公共有效区域裁剪、融合、清晰度评价和结果保存。三种算法使用相同的配准输入和评价区域，并分别记录融合耗时、实际参数及失败原因。输入为外部已采集的照片，原始文件只读，所有结果保存到 `results/`。

## 三种算法

| 算法 | 命令行名称 | 基本思路 |
|---|---|---|
| 局部 Laplacian（Baseline） | `laplacian` | 用局部 Laplacian 响应衡量清晰度，选择最清晰的来源帧，经多数投票平滑来源图后重建彩色图像。 |
| GFDF 引导滤波融合 | `guided_filter` | 检测清晰区域，通过引导滤波细化清晰度图、去除小区域，再生成权重进行彩色融合。 |
| Laplacian 金字塔融合 | `laplacian_pyramid` | 将图像分解为多尺度金字塔，细节层按区域能量选择，顶层结合局部方差和熵融合，最后逐层重建。 |

GFDF 依据 Qiu 等（2019）的[论文](https://doi.org/10.1016/j.image.2018.12.004)及[作者 MATLAB 代码](https://github.com/bitname/Multi-focus-image-fusion-GFDF)（提交 `40c2f25`）；金字塔方法依据 Wang、Chang（2011）的[论文](https://www.jcomputers.us/vol6/jcp0612-07.pdf)。两者均做了多帧联合处理、彩色及 16 位输入适配；GFDF 另统一了并列选择规则，并对权重裁剪、归一化。具体差异和参数说明见对应算法文件。

## 数据与项目结构

```text
focus-stacking/
├── data/raw/<tool>/
│   ├── view1/                 # 一个独立焦点序列，多张不同焦平面的照片
│   └── view2/                 # 另一个视角的独立序列
├── methods/                  # 三种融合算法
├── evaluations/              # 清晰度指标；自动伪影评价仍为占位
├── results/                  # 融合图像和实验报告
├── run_benchmark.py          # 统一运行入口
└── requirements.txt          # Python 依赖
```

- 视角目录使用小写 `view` 加无前导零的正整数，例如 `view1`、`view3`、`view10`。程序按数字排序，编号可以不连续。
- 每个视角目录直接存放一个序列的图像，不嵌套子序列。支持 PNG、JPEG、TIFF、BMP，以及三通道 `uint8` / `uint16` 彩色图像；同一序列的尺寸和位深须一致。
- 图像按文件名自然排序，例如 `frame_2.png` 在 `frame_10.png` 之前。文件名应与预期焦平面顺序一致；程序不推测缺失的采集步长。

## 安装与运行

需要 Python 3.10 或更新版本。以下命令在项目根目录的 PowerShell 中执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

将示例中的 `tool_01` 替换为实际刀具目录名：

```powershell
# 处理该刀具的所有视角，默认运行全部算法并评价
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01" --method all
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Kro_DEMO_10" --method all
# 仅处理一个视角，分别选择三种算法
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01\view1" --method laplacian
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01\view1" --method guided_filter
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01\view1" --method laplacian_pyramid

# 一次处理多个刀具或视角
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01\view1" "data\raw\tool_02"

# 查看完整命令帮助
.venv\Scripts\python.exe run_benchmark.py --help
```

`--input` 必填，仅接受 `data/raw/` 下的刀具目录或编号视角目录；重叠输入中的同一视角只处理一次。省略 `--method` 时等同于 `--method all`。

常用可选参数：

| 参数 | 作用 |
|---|---|
| `--window-size 21` | Baseline 的局部清晰度窗口边长，默认 21 像素，须为正奇数。 |
| `--smooth-size 5` | Baseline 的多数投票窗口边长，默认 5 像素，须为正奇数；设为 1 关闭平滑。 |
| `--roi X Y WIDTH HEIGHT` | 指定清晰度评价矩形，坐标基于配准裁剪后的输出图；默认评价整个有效区域。指定时额外生成 `roi_preview.png`。 |
| `--registration-max-size 1600` | 配准特征估计的最长边上限，默认 1600 像素，输出保留原始像素尺度并裁剪公共有效区域。 |

GFDF 和金字塔方法的默认参数集中在 [guided_filter.py](methods/guided_filter.py) 和 [laplacian_pyramid.py](methods/laplacian_pyramid.py) 顶部的 `DEFAULT_PARAMETERS`。修改后重新运行即可生效，参数的范围、来源和影响见代码注释。GFDF 默认值来自论文和作者代码；金字塔默认层数 5、能量窗口 5×5、顶层窗口 3×3、熵灰度级数 256 为工程选择，其中能量窗口等参数经 `Kro_DEMO_10` 数据检查，不代表其他刀具上的最优值。实际生效参数写入 `run_info.json`。

也可以扫描金字塔参数，例如比较不同层数：

```powershell
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01\view1" --method all --pyramid-sweep levels 3 4 5 6 7
```

扫描支持 `levels`、`min_top_size`、`top_window`、`entropy_bins`，以及能量窗口边长 `energy_size`（二项式权重）和 `energy_box`（均匀权重）。扫描值覆盖对应默认参数，结果保存到该视角的 `sweep_<参数名>/` 子目录。

GFDF 参数用 `--guided-sweep` 扫描，例如比较不同的去小区域阈值：

```powershell
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\tool_01\view1" --method all --guided-sweep small_region_ratio 0 0.001 0.005 0.01
```

可扫描 `guided_filter.py` 顶部的全部 6 个参数：`mean_window`、`focus_radius`、`weight_radius` 取整数，`focus_eps`、`weight_eps`、`small_region_ratio` 可取小数。

两种扫描的共同规则：
- 每次只扫描一个参数，`--pyramid-sweep` 和 `--guided-sweep` 不能同时使用。
- 必须选中被扫描的方法（`--method` 为该方法或 `all`）。用 `all` 时，其他方法按默认参数各运行一次，作为参照。
- 每个取值生成一张图，图名为 `<方法>_<参数名><值>.png`，例如 `guided_filter_small_region_ratio0.005.png`。
- 同一参数重新扫描时，先删除上次留下的扫描图。
- 取值非法或重复时直接报错。`run_info.json` 的 `sweep` 记录方法、参数和取值，`parameters` 按图名记录每个取值实际生效的参数。

## 输出与评价

常规运行结果保存在 `results/<tool>/view<N>/`：

```text
laplacian.png             # Baseline 融合图
guided_filter.png         # GFDF 融合图
laplacian_pyramid.png     # 金字塔融合图
benchmark.csv             # 图像数量、清晰度指标、融合耗时、状态和错误信息
metrics_comparison.png    # 清晰度指标对比图
run_info.json             # 输入顺序、实际参数、配准信息及软件版本
roi_preview.png           # 仅在指定 --roi 时生成：参考帧原图上的 ROI 红框
```

`roi_preview.png` 的底图是配准参考帧（中间帧）的未处理原图，可与融合图对比配准裁剪前后的差异。参考帧不做变换，因此 ROI 坐标加上裁剪偏移（`run_info.json` 中 `registration.crop_xywh` 的前两个值）后与原图精确对应。红框画在 ROI 外侧，框内像素与原图一致；ROI 贴近原图边缘时，该侧红框会被截掉。`run_info.json` 的 `roi_preview` 记录底图文件名、原图坐标下的 ROI 及可能的生成错误；预览图生成失败不影响融合和评价。未指定 `--roi` 时不生成预览图，并删除上次运行留下的旧预览图。

仅生成本次成功方法的融合图，保留输入位深。重复运行会替换所选方法的旧图和当前目录的报告；失败方法的旧图会被移除。

当前清晰度指标为 **Tenengrad** 和 **Laplacian 方差**，分别报告，不合成总分。噪声和接缝也可能提高分数，应结合刀刃、前刀面纹理和高光区域的局部目视检查。自动接缝与伪影评价尚未实现。融合耗时只覆盖各算法的完整 `stack()` 调用，公共读取、配准、评价和保存不计入其中。

原始数据与生成结果默认由 `.gitignore` 排除；正式通过 Git 交付实验时，需明确选择要纳入版本管理的结果文件。

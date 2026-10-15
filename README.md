# SG3 Focus Stacking

本项目处理不同焦平面的刀具图像，通过焦点堆叠扩展景深。当前已实现局部 Laplacian Baseline、公共配准、融合图评价和结果报告。Method A、Method B 尚未确定，保留明确占位。

仅处理外部提供的图像，不包含采集、相机控制、机床控制、LED 控制或显微镜验证。代码中的注释和文档字符串使用英文。

## 安装与运行

需要 Python 3.10 或更新版本：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Process all three views with all methods (the default).
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Kro_DEMO_10" --method all

# Process only one view with the Laplacian baseline.
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Kro_DEMO_10\face" --method laplacian

# Adjust focus and smoothing windows.
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Kro_DEMO_10\face" --window-size 11 --smooth-size 3

# Evaluate a rectangle in the cropped output: x y width height.
.venv\Scripts\python.exe run_benchmark.py --input "data\raw\Kro_DEMO_10\face" --method laplacian --roi 500 300 800 600
```

本次实现已在本地创建 `.venv` 并安装依赖，可直接使用其中的 Python。普通 `python` 命令是否可用取决于本机 PATH。

`--input` 必填，接受项目 `data/raw/` 下的刀具目录或其 `flank`、`rake`、`face` 目录。相对路径以项目根目录为基准，也支持绝对路径。更换刀具只需修改该参数。

`--method` 支持 `laplacian`、`method_a`、`method_b`、`all`，默认 `all`。`--smooth-size 1` 关闭来源图平滑。窗口必须为正奇数。

## 输入约定

```text
data/raw/<tool>/flank/
data/raw/<tool>/rake/
data/raw/<tool>/face/
```

- 每个视角目录包含一个独立焦点序列；只读取该目录中的图像，不递归合并子序列。
- 支持 PNG、JPEG、TIFF、BMP；输入必须是三通道彩色 uint8 或 uint16，同一序列尺寸和位深一致。
- 文件按自然文件名顺序排序，例如 `frame_2.png` 先于 `frame_10.png`。请确保文件名顺序符合预期焦平面顺序；程序不推测缺失的采集元数据。
- 不静默跳过损坏图像；读取失败会记录原因。原始文件不会被覆盖、修改或删除。

## 公共处理与算法

`run_benchmark.py` 是唯一主要执行入口，并包含 `align_images()`、`to_gray_images()` 两个公共函数。

配准以排序后的中间帧 `len(images) // 2` 为参考，使用 SIFT 特征匹配和 RANSAC 筛选对应点，再拟合统一缩放与二维平移。先尝试直接匹配参考帧；若失败，则匹配靠近参考帧方向、已成功配准的相邻帧，再组合变换到中间帧。相邻匹配同样必须通过可靠性检查，其使用情况和直接匹配失败原因保存到 `run_info.json`。组合变换可能累积误差，仍需人工检查细节对齐。

最终变换不包含旋转、剪切或透视变换。特征估计默认将最长边缩小到不超过 1600 像素，输出仍保留参考帧的原始像素尺度。

可靠匹配少于 6 个、拟合残差过大、尺度不在 0.5–2.0 或公共有效区域少于原图面积的 25% 时，明确记录配准失败。强失焦或缺少纹理的序列可能无法配准，不自动假定已经对齐。所有帧使用线性插值，并共同裁剪到插值不包含黑色填充的有效矩形。

灰度副本先按位深上限归一化到 `[0, 1]`，再转为 float32 灰度，不对每张图独立做最大最小值归一化。

`methods/laplacian.py` 负责：

1. 计算 `ksize=3` 的有符号 Laplacian 响应。
2. 对响应绝对值计算默认 21×21 的局部均值。
3. 每个位置选择分数最高的来源编号；同分时保留靠前的输入帧。
4. 对编号图进行默认 5×5 多数投票；票数并列时优先保留原编号，否则选较小编号。
5. 按来源编号选择整组三个 BGR 通道，返回融合彩图，不做边界加权混合。

入口将配准彩色图和灰度副本设置为只读，复用于所有算法。评分与投票按帧计算，避免分配完整的 N×H×W 清晰度体积，但仍需在内存中保留彩色图和灰度副本。

## 评价与计时

`evaluations/sharpness.py` 只评价融合图，输出两个独立数值：

- `tenengrad`：3×3 Sobel 水平和垂直梯度平方和的均值，不设阈值。
- `laplacian_variance`：`ksize=3` 有符号 Laplacian 响应的总体方差，`ddof=0`，不先取绝对值。

导数计算使用 `BORDER_REFLECT_101`。评价排除图像最外侧一圈像素，减少图像边界处理的影响。默认评价整个裁剪后的有效区域；使用 `--roi` 可指定刀具区域，其坐标以裁剪后的融合图为准。同一次运行的所有方法使用同一个 ROI。

暂不评价输入图、不计算融合前后变化率、不合成总分。噪声和接缝也可能提高分数，因此应结合固定局部区域的人工检查。自动伪影评价继续在 `evaluations/artifacts.py` 中保留占位，人工记录放在 `results/`。

`fusion_seconds` 仅测量 `stack()` 的调用时间，包括清晰度计算、来源选择、平滑和彩色融合，不包含读取、配准、公共灰度转换、评价、保存或绘图。公共准备总耗时另存于 `run_info.json`。

## 输出

```text
results/<tool>/<view>/
├── laplacian.png
├── benchmark.csv
├── metrics_comparison.png
└── run_info.json
```

方法成功后才生成对应 `<method>.png`，保留输入位深。每次运行替换所选方法的旧结果及该视角的报告；若该方法此次失败，其旧结果图会移除，以免误认为新结果。

`benchmark.csv` 使用 UTF-8 BOM，每个选定方法一行，只包含以下字段：

```text
tool,view,method,image_count,tenengrad,laplacian_variance,fusion_seconds,status,error
```

CSV 不包含 `input_files`、`parameters`、`focus_step`。输入顺序、运行参数、配准参考帧、裁剪区域、变换及软件版本单独保存在 `run_info.json`，用于复查，不推测未知焦平面步长。

`metrics_comparison.png` 将两个指标分别画在并排的柱状子图中。只有一个成功方法也生成图；失败或尚未实现的方法不画零分柱；没有有效评分时明确显示 `No valid scores`。

状态为 `success`、`failed`、`not_implemented`。失败信息包含阶段与原因；无效评分留空。单个视角失败后继续处理其他视角。存在实际失败或没有任何成功方法时，退出码为 1；成功且仅有候选方法占位时为 0。指定无效 CLI 参数时由 argparse 返回退出码 2。

## 验证与交付

```powershell
.venv\Scripts\python.exe -m unittest -v test_focus_stacking
```

测试覆盖互补清晰区域、输入不被修改、16 位输出、来源平滑、指标公式、已知缩放和平移、三视角遍历、单方法绘图、占位与失败报告，以及重跑后避免旧结果误用。合成测试输入均放在 `results/` 的临时目录中，不写入真实 `data/raw/`。

保持现有简单结构，不增加算法专属运行脚本。原始输入和生成结果默认被 `.gitignore` 排除；正式 Git 交付时明确选取结果文件，避免遗漏。当前工作目录没有 `.git`，本次实现不自动初始化仓库或创建提交。

公式与研究方案见 [Laplacian-Based Focus Stacking Plan.md](Laplacian-Based%20Focus%20Stacking%20Plan.md)。配准基础操作参考 [OpenCV 特征匹配文档](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)，滤波操作参考 [OpenCV 图像滤波文档](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)。

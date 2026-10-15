# SG3 Focus Stacking — Agent Instructions

## Project Purpose

本仓库是 SG3 专属的 Focus Stacking 图像处理、算法比较与 Benchmark 项目。核心目标是使用在不同焦平面采集的刀具图像，通过焦点堆叠扩展弯曲刀具前刀面的景深，并系统比较不同算法的表现。

## Scope

- 仅处理已经由外部系统提供的图像数据，原始焦点序列直接放入 `data/raw/`。
- 一个焦点序列包含多个不同焦平面的图像；后续输入可能包含多个刀具、多个角度和多个独立焦点序列。
- 不负责相机控制、Basler SDK、机床控制、机床运动路径、图像采集程序或 LED 控制。
- 不使用显微镜数据、显微镜真值或显微镜对照组，不涉及显微镜验证。

## Project Structure

- `data/raw/`：已经采集完成的原始焦点序列，是只读输入。算法代码不得覆盖、修改或删除原始文件。
- `methods/`：不同焦点堆叠算法的实现，不负责独立组织完整 Benchmark。
- `evaluations/`：评价指标实现；输入图像或实验结果，返回对应评价值，不遍历整个数据集、不组织完整实验。可以按研究需要增加指标文件。
- `results/`：统一保存融合图像、Benchmark CSV、评价指标、运行时间、失败记录和实验图表。
- `run_benchmark.py`：唯一主要执行入口，后续统一负责读取焦点序列、选择算法、调用融合与评价、记录运行时间、保存结果及汇总 Benchmark。

## Architecture Rules

- 保持项目简单，优先在现有结构中解决问题。
- 不主动增加 `src/`、`preprocessing/`、`configs/`、复杂 Python package 或额外抽象层。
- 不主动拆出大量运行脚本，不增加 `run_stacking.py`、`run_evaluation.py` 或各方法专属运行脚本。
- `run_benchmark.py` 是统一执行入口；默认依次运行全部方法并评价，支持 `--method laplacian`、`--method method_a`、`--method method_b` 和 `--method all`。
- 算法实现放在 `methods/`，评价指标放在 `evaluations/`。
- 不同算法尽量采用统一接口：`stack(images, **kwargs)` 返回融合图像。算法不得修改传入的原始图像数据。
- 原始文件不得修改，生成结果只能写入 `results/`，不得写回 `data/raw/`。
- Benchmark 中所有方法必须尽可能采用相同输入及一致的数据、评价和计时流程，保证比较公平。记录方法参数、输入选择及失败情况，使结果可复查。
- 用户已明确授权按 `Laplacian-Based Focus Stacking Plan.md` 实现基础算法及统一 Benchmark；不添加 GUI、数据库、Docker、CI/CD 或采集控制代码。

## Current Methods

- `methods/laplacian.py`：Baseline，采用局部 Laplacian 清晰度、来源选择、多数投票平滑和彩色像素重建；已进入实现阶段。
- `methods/method_a.py`：第一种候选焦点堆叠方法，占位，算法尚未确定。
- `methods/method_b.py`：第二种候选焦点堆叠方法，占位，算法尚未确定。

除非用户之后明确要求，不得自行决定 Method A 或 Method B 的具体算法。

## SG3 Research Requirements

1. 自行实现简单、可解释的焦点堆叠 Baseline；目前没有现成 Python baseline，暂定采用基于局部 Laplacian 清晰度的焦点选择。
2. 在 Baseline 之外至少实现两种候选焦点堆叠方法。
3. 所有方法尽可能使用相同的输入和评价流程，保证 Benchmark 公平。
4. 系统比较清晰区域覆盖、细节保留、融合接缝、融合伪影、算法运行时间和失败情况。
5. 研究输入照片数量和焦平面采样步长对融合结果的影响。记录采样选择和可获得的步长信息，不臆测缺失的采集元数据。
6. 正式机床侧验证数据至少包含 5 把刀具，每把至少 50 张图像，总计至少 250 张图像。数据来自多个刀具角度或多个独立焦点序列，而不是单个超长序列。此要求描述输入数据，不能据此加入采集或机床控制程序。
7. 不使用显微镜数据、显微镜真值或显微镜对照组。
8. 没有全清晰真值图像时，不应把 SSIM 当作主要绝对评价指标。
9. 最终保存代码、融合结果、评价数据和实验图表，并通过 Git 交付。当前 `.gitignore` 默认排除原始数据和生成结果；正式交付时明确选择需要版本管理的结果文件，不应因忽略规则遗漏交付内容。

## Development Principle

Agent 修改项目时应优先在当前简单结构内解决问题，而不是通过增加新的目录、抽象层或脚本解决问题。如果未来确实需要增加目录或大规模调整架构，应先说明必要性，不得自动重构。

当前任务已由用户推进为基础方案实现。Method A、Method B 和自动伪影评价仍保留占位；占位函数应明确表示尚未实现，不返回虚假的融合图像或评价值，入口不得将占位状态报告为已完成的实验。

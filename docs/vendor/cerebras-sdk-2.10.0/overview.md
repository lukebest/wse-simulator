# Cerebras SDK 2.10.0 — 概览

## 是什么

Cerebras SDK 是 **Wafer-Scale Cluster 上的 kernel 开发框架**：用 **CSL（Cerebras Software Language）** 编写运行在每个 PE 上的程序，用 **SdkLayout** 在编译期放置 tile、配置 color 路由，用 **SdkRuntime** 在 host 侧加载程序、启动函数、通过 memcpy 流传输 tensor。

官方入口：[Documentation for Developing with the Cerebras SDK](https://sdk.cerebras.net/)

## 软件栈组件（2.10.0）

| 组件 | PyPI | 说明 |
|------|------|------|
| `cerebras-sdk` | [2.10.0](https://pypi.org/project/cerebras-sdk/2.10.0/) | SDK 元包 / 入口 |
| `cerebras-appliance` | ==2.10.0 | 集群 appliance 运行时 |
| `cerebras-pytorch` | [2.10.0](https://pypi.org/project/cerebras-pytorch/2.10.0/) | PyTorch 训练栈（独立文档 [docs.cerebras.net](https://docs.cerebras.net/)） |

支持：support@cerebras.net · Discord：https://discord.gg/ZqvYS2e2rY

## 安装

SDK 通常在 **Singularity 容器** 内使用（见官方 [Installation and Setup](https://sdk.cerebras.net/)）。PyPI 安装示例：

```bash
pip install cerebras-sdk==2.10.0
```

实际 kernel 编译、仿真与上板运行需配合 Cerebras ML Software 2.10 环境。

## Host API（Python）

| API | 文档 | 职责 |
|-----|------|------|
| **SdkRuntime** | [Host Runtime](https://sdk.cerebras.net/) | 加载 CSL 程序、调用 exported 函数、host↔device 数据 |
| **SdkLayout** | [SdkLayout API](https://sdk.cerebras.net/) | 多 region layout、`@set_tile_code`、`@set_color_config` 等编译期配置 |
| **SDK Appliance API** | [Running on Cluster](https://sdk.cerebras.net/) | 在 Wafer-Scale Cluster 上编译/运行 |

2.10.0 新增/增强的 layout 选项包括：

- `SdkLayout` 的 `cslc_prefix`
- `SdkLayout.compile` 的 `f16_type`（`F16` / `BF16` / `CB16`）
- `SdkLayout.compile` 的 `libs`（额外库搜索路径）

## 开发与调试

- **CSL 语言与编译器**：[CSL Language Guide](https://sdk.cerebras.net/csl/language/)
- **教程**：GEMV 系列（Topic 1–16）、pipeline 示例
- **SDK GUI**：fabric 执行、wavelet trace、PE 状态可视化（2.10.0 起 WSE-3 支持 instruction traces）
- **示例仓库**：[sdk-examples](https://github.com/Cerebras/sdk-examples)（官方引用）

## 与本仓库 `wsesim` 的关系

| SDK 概念 | `wsesim` 对应 |
|----------|---------------|
| `@set_color_config` 的 `rx`/`tx` 方向 | `ColorPlan.dest[node][color]` 方向总线 |
| `@get_color(0..23)` | `color_catalog.py` 24 色 catalog |
| wavelet-triggered data task | `Color.task = color_id * 4` |
| `@activate` / `@block` / `@unblock` | 仿真中 active/block 语义（见 `color_usage_guide.md` §4A） |
| `@initialize_queue(..., .color = ...)` (WSE-3) | input queue ↔ color 绑定 |

专利 US10,515,303 描述的 Router RR / 反压 / Picker 细节见 `docs/color_mechanism_analysis.md`；SDK 文档从 **程序员配置面** 描述 color，两者互补。

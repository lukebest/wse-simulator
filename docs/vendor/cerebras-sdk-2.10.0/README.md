# Cerebras SDK 2.10.0 — 参考摘要

本目录收录 **Cerebras Wafer-Scale Cluster SDK 2.10.0** 的公开文档摘要，供 `wsesim` 对照片上 fabric / color 语义使用。

> 来源：官方文档 [sdk.cerebras.net](https://sdk.cerebras.net/)（2026-03-15 发布 2.10.0 文档）、PyPI 包元数据。  
> 本仓库 **不 vendoring** SDK 源码；在集群/Singularity 容器内通过 `pip install cerebras-sdk==2.10.0` 安装。

## 与 Cloud REST SDK 的区别

| 产品 | PyPI / 文档 | 用途 | 是否暴露 Color/Fabric |
|------|-------------|------|----------------------|
| **Cerebras SDK** | `cerebras-sdk` 2.10.0 · [sdk.cerebras.net](https://sdk.cerebras.net/) | 片上 kernel 开发（CSL）、SdkRuntime、layout、memcpy | **是** — `@get_color`、`@set_color_config`、wavelet-triggered tasks |
| **Cerebras Cloud SDK** | `cerebras-cloud-sdk` · [GitHub](https://github.com/Cerebras/cerebras-cloud-sdk-python) | 云端推理 REST（chat / completions / models） | **否** |
| **Cerebras PyTorch** | `cerebras-pytorch` 2.10.0 · [docs.cerebras.net](https://docs.cerebras.net/) | 集群训练 / 推理框架 | 间接（编译器生成路由，非手写 CSL） |

`wsesim` 建模的是 **SDK / 专利层面的片上 NoC color 机制**，不是 Cloud REST 客户端。

## 文档索引

| 文件 | 内容 |
|------|------|
| [overview.md](overview.md) | 软件栈、PyPI 包、安装、Host API |
| [fabric-and-color.md](fabric-and-color.md) | Color、路由、WTT、activate/block — 与 `wsesim` 对照 |
| [release-notes-2.10.0.md](release-notes-2.10.0.md) | 2.10.0 发布说明摘要 |
| [official-links.md](official-links.md) | 官方文档 URL 索引 |

## 版本信息

- **SDK 版本**：2.10.0（2026-03-15 文档；PyPI wheel 2026-03-10）
- **前序版本号**：1.4.0 → 2.10.0（编号方案与 Cerebras ML Software 对齐）
- **兼容**：Cerebras Wafer-Scale Cluster + Cerebras ML Software 2.10
- **依赖**：`cerebras-sdk` → `cerebras-appliance==2.10.0`

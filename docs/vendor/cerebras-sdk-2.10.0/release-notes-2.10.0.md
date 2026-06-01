# Cerebras SDK 2.10.0 — 发布说明摘要

> 完整原文：[SDK Release Notes](https://sdk.cerebras.net/sdk-release-notes/sdk-rel-notes-cumulative) · 文档更新 2026-03-15

## 版本与兼容

- **发布日期**：2026-03-15（文档）；PyPI `cerebras-sdk` 2.10.0：2026-03-10
- **版本号变更**：1.4.0 之后改为与 **Cerebras ML Software** 同号（2.10）
- **测试环境**：Wafer-Scale Cluster + **Cerebras ML Software 2.10**

## CSL 语言与编译器（节选）

与 fabric / 通信相关的新增：

- **`sr` 类型**与 `@get_sr`、`@load_to_dsr_xdsr_sr`（mem4d_dsd 必须配 XDSR+SR）
- **FIFO** `@allocate_fifo` 的 `.full_action` / `.empty_action`（WSE-3：fault、suspend 等）
- **`@bind_rotating_tasks`**：task rotation
- **WSE-3 dense mode** `@initialize_queue`
- **`circbuf_dsd`**（comptime，经 `@load_to_dsr_xdsr` 加载）
- **`tile_config.filters`**：half-wavelet filters（WSE-3）
- **DSD 限制**：WSE-3 上 fabin_dsd 操作数至多一个；部分 operand 组合扩展
- **SIMD**：WSE-2 默认 `simd_32`；WSE-3 有效选项 `simd_off` / `simd_max`

其他：任意位宽整数 `iN`/`uN`、`packed struct`、union、widening coercion、`anyopaque`、新标准库（`bit`, `fmt`, `meta`, `math.subsat`）等。

## SdkRuntime / SdkLayout

- `SdkLayout`：`cslc_prefix`
- `SdkLayout.compile`：`f16_type`（F16/BF16/CB16）、`libs` 搜索路径

## 示例与工具

- 新教程：**Topic 16 — Reusing Output Queues on WSE-3**（`@queue_flush`）
- **SDK GUI**：WSE-3 支持 instruction traces

## 已修复（节选）

- WSE-3 instruction traces
- `tile_config.teardown.exit()` RAW hazard
- DSD 作为 comptime 参数、 `@set_dsd_base_address` 等编译器 bug

## 与本仓库

2.10.0 文档强化了 **WSE-3 queue/color 分离** 与 **filters / rotating tasks**，但不改变 color 0–23 作为静态路由 VN 的核心模型。`wsesim` 当前以专利 + catalog 为主轴；WSE-3 queue 细节可在后续仿真中扩展。

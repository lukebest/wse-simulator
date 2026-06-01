# Fabric 与 Color（Cerebras SDK 2.10.0）

> 摘要自 [Task Identifiers](https://sdk.cerebras.net/csl/language/task-ids)、[Builtins](https://sdk.cerebras.net/csl/language/builtins)、教程 Topic 3/8 等。  
> 与 `wsesim` 及 `docs/color_usage_guide.md` 对照阅读。

## Color 是什么

硬件将 **ID 0–23** 识别为 PE 间传递 wavelet 的 **虚拟通信通道（routable colors）**。编译期通过 layout / comptime 为每个 PE 的每条 color 配置 **静态路由**；运行时不在 SDK 层做动态路由决策。

```csl
const c = @get_color(4);
```

## WSE-2 vs WSE-3

| 方面 | WSE-2 | WSE-3 |
|------|-------|-------|
| Data task ID | 由 **color** 构造：`@get_data_task_id(@get_color(n))` | 由 **input_queue** 构造：`@get_data_task_id(iq)` |
| Queue ↔ color | 隐式绑定 color | `@initialize_queue(iq, .{ .color = c })` 显式绑定 |
| Fabric DSD | `fabric_color` 字段 | 多通过 `input_queue` / `output_queue` |
| 其他 | — | dense mode queue、`circbuf_dsd`、FIFO full/empty actions、task rotation |

## Wavelet-Triggered Task（WTT）

**Data task**：某 color（WSE-2）或某 input queue 所绑 color（WSE-3）上到达 wavelet 时 **激活**，Picker 选中后执行。

```csl
const main_task_id: data_task_id =
  if (@is_arch("wse2")) @get_data_task_id(@get_color(0))
  else if (@is_arch("wse3")) @get_data_task_id(h2d_iq);

task main_task(data: i16) void { /* ... */ }

comptime {
  @bind_data_task(main_task, main_task_id);
  if (@is_arch("wse3")) {
    @initialize_queue(h2d_iq, .{ .color = @get_color(1) });
  }
}
```

**Control task**：由 control wavelet 的 payload 中的 control task id 触发；关联 color 需 `@unblock` 后才能被调度。

### 激活条件（与专利 Picker 一致）

Data task 可执行当且仅当：

1. 绑定的 data task id **已激活**（wavelet 到达，或 `@activate`）
2. 对应 color **未 block**（`@block` / `@unblock`）

## 路由配置：`@set_color_config`

在 **layout** 中为指定 PE 的指定 color 设置路由（编译期静态）：

```csl
layout {
  @set_color_config(0, 0, data_color, .{
    .routes = .{ .rx = .{ WEST }, .tx = .{ RAMP, EAST } }
  });
}
```

- **`rx`**：允许从哪些方向 **接收**（`EAST` / `WEST` / `NORTH` / `SOUTH` / `RAMP`）
- **`tx`**：允许向哪些方向 **发送**（可多出口 → 静态多播，对应专利 Dest 661）
- 也可用 10-bit **bitvector** 编码 routes（见 `directions` 库）
- **`filter`**：按 wavelet 内容筛选是否转发到 CE（range filter 等）
- **`switches`**：pos1/pos2/pos3 路由切换（control wavelet 驱动）

`@set_local_color_config`：在 PE 的 comptime 块中设置本 PE 的 color，无需坐标。

⚠️ 同一 PE + 同一 color 只能配置一次。多 `rx` 方向同时收 wavelet 行为未定义。

## activate / block / unblock

| Builtin | 作用 |
|---------|------|
| `@activate(id)` | 启动时或 comptime 激活 task（local task） |
| `@block(id)` | 阻止该 color 上的 wavelet 激活 task；使用 fabin DSD 前常需 block 该 color |
| `@unblock(id)` | 解除 block |

这与专利 FIG 9B/9C 的 Active/Block 位语义一致；`wsesim` 在文档层建模，不在每个 cycle 仿真 CE 指令。

## Fabric DSD 与 memcpy 色

Host↔device 流使用 memcpy 模块分配的 **streaming colors**（如 `MEMCPYH2D_DATA_1`）。memcpy 模块负责这些色的路由；程序员绑定 task 或 DSD 即可。

使用 **fabin_dsd** 消费向量时，应 **block** 该 color，否则空 task 会先消费 wavelet（见 Pipeline 1 教程）。

## Filters（Topic 8）

`@set_color_config(..., .filter = ...)` 可配置 range filter 等，仅将匹配 index 范围的 wavelet 送入 CE。用于多 PE 共享一色但各 PE 只处理子集。

## 与 `wsesim` 24 色 catalog

SDK 程序员 **按应用** 分配 0–23，无全局固定「bcast_row_east = 2」目录。本仓库 `color_catalog.py` 的 24 名是 **集合通信仿真约定**，映射到 SDK 能力：

- 每个 catalog 色 ≈ 一次 `@set_color_config` 的 `tx`/`rx` 方向集
- `unicast_xy/yx` ≈ 未填 dest、按 dst 逐 hop 的模式
- `ColorBudget` MINIMAL/COMPACT 用 unicast 色时间复用，对应 SDK 允许的非重叠 phase 复用同一 color

硬件调度（Router RR、反压、多出口多播）见 `docs/color_usage_guide.md` §4A（专利推导）。

## 官方页面

- Task IDs: https://sdk.cerebras.net/csl/language/task-ids
- Builtins (`@get_color`, `@set_color_config`, …): https://sdk.cerebras.net/csl/language/builtins
- Topic 3 Streaming Wavelet: https://sdk.cerebras.net/csl/code-examples/tutorial-topic-03-streaming-wavelet-data
- Topic 8 Filters: https://sdk.cerebras.net/csl/code-examples/tutorial-topic-08-filters

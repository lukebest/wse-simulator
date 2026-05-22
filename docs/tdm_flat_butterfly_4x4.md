# 4x4 TDM Flattened Butterfly Example

本文档给出 4x4（16 节点）物理 mesh 上的 TDM flattened butterfly 示例，覆盖 `(k,n)=(4,2)` 与 `(2,4)`，并与不做时分的 `mesh2d` 包交换做集合通信对比。

## 双层模型

- 物理层：`Mesh2D(rows=4, cols=4)`，16 个节点。
- 逻辑层：`k^n=16` 的 FB 逻辑长链路。
- 映射：每条逻辑长链路按 XY DOR 映射到物理 mesh 有向边路径。

## 全局 color 时分

- 对所有逻辑长链路做全局边着色，得到 `ColorPlan`：
  - `C`：周期长度
  - `color_of_logical[(u,v)]`：逻辑长链路 color
  - `link_active[(p_src,p_dst)][c]`：物理有向边在 color `c` 是否激活
- 周期 `t` 当令 color 为 `t mod C`。
- flit 若当前节点发出方向不匹配其 color，则等待到目标 color 后再发送。

## 集合通信映射

- allreduce：`nd_dimension_exchange_allreduce`
- allgather：`nd_dimension_exchange_allgather`

基线 mesh2d：

- allreduce：`2d_ring`
- allgather：`direct_allgather`

## 仿真输出

脚本：`examples/run_tdm_flatbf_4x4.py`

- 拓扑：`mesh2d_4x4_ps`、`tdm_fb_k4_n2`、`tdm_fb_k2_n4`
- 集合：`allreduce`、`allgather`
- 消息：`1KB/16KB/256KB`
- 输出：`outputs/tdm_flatbf_4x4/results.csv`

## 着色可视化（如何读图）

脚本：`examples/visualize_tdm_coloring_4x4.py`

输出目录：`outputs/tdm_flatbf_4x4/coloring/`

每个 `(k,n)` 有三类图：

| 文件 | 含义 |
|------|------|
| `*_overview.png` | **物理 mesh 负载图**。圆圈内数字是 **节点 id**（行优先 0–15）。边上数字是 **load(e)**：有多少条逻辑 FB 长链路会经过这条物理边。**不是 color 编号**。 |
| `*_colors.png` | **每个 color 一张子图**。弯曲箭头 = 该 color 当令时激活的一条 **逻辑 FB 长链路**（`src->dst, dim`）。同一 color 内箭头互不共享物理边。 |
| `*_guide.png` | **导读示例**：(a) 一条逻辑链路 → (b) 它在 mesh 上的 XY 物理路径 → (c) 它属于哪个 color。 |
| `*_summary.txt` | 文字版：每个 color 有多少条逻辑链路、前几条样例。 |

节点编号（4×4 行优先）：

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

以 `k=4,n=2` 为例：dim0 是“同行 4 节点全连”，dim1 是“同列 4 节点全连”。16 条逻辑长链路被分成 `C=4` 个 color，每个 cycle 只激活其中一个 color 的子集。

## 复杂度和观察点

- `C` 越大，单流等待 color 的平均延迟越高，但冲突可控。
- `(4,2)` 更高 radix，通常 `C` 更高；`(2,4)` stage 更多但单 stage 冲突更低。
- 重点对比指标：`makespan_cycles`、`avg_latency`、`avg_link_util`、`total_flits`。


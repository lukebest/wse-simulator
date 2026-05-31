# Color 用法与规则指南

本文档说明 Cerebras WSE 上 **Color（虚拟网络）** 的语义、规则，以及在各类集合通信场景下如何用 color 传输数据。实现代码位于 `wsesim/network/color*.py`；硬件依据为专利 **US10,515,303**。

---

## 1. 与 `cerebras-cloud-sdk-python` 的关系

| 层次 | 路径 | 是否包含 Color |
|------|------|----------------|
| **Cloud REST API** | `vendor/cerebras-cloud-sdk-python/src/cerebras/cloud/sdk` | **否** — 仅为 HTTP 客户端（chat completions、models） |
| **片上 NoC / 编译期路由** | `wsesim/network/` + 本文档 | **是** — 仿真与文档建模 |

Cloud SDK README 描述 WSE-3 为「全球最大 AI 处理器」，但 **SDK 源码不包含 wavelet、fabric、color、collective 等片上网络 API**。Color 是编译器/运行时根据 NN 通信图在 **编译期** 写入每个 PE 的静态转发表；本仓库的 `wsesim` 是对该机制的 cycle-accurate 仿真。

> 若未来 Cerebras 发布 wafer SDK / fabric API，应对照本文 catalog 与 `color_usage.py` 中的 `COLLECTIVE_SCENARIOS` 做映射。

---

## 2. Color 是什么

**Color = 叠加在物理 2D mesh 上的一条虚拟网络（VN）**，具有：

1. **独立 per-color 输入缓冲**（Router 600 的 Data Queues 650，默认 2 entries × num_colors）
2. **共享物理链路** — 不同 color 的包可在不同链路上并行，同链路则时分
3. **固定静态路由** — `dest[node][color] → {next_hop}` 在编译期确定，运行时 **不** 根据 dst 重算（方向总线类 color）
4. **独立 per-color 反压**（Stall Out/In）— 沿数据流反向传播

### 2.1 Wavelet 中的双重角色

| 角色 | 含义 |
|------|------|
| **VN 选择器** | 包 header 的 color 字段选中一条 VN → 使用该 color 的 Dest 661 转发表 |
| **Task 选择器** | 到达 CE 后 `instruction_addr = base + color × 4`，触发对应计算任务 |

仿真中：`Color.task = color_id * 4`（见 `color_catalog.py`），`ColorNetwork.send_packet` 会自动写入 `packet.task`。

### 2.2 规模无关的 Color 目录

**Color 的语义 ID 与 mesh 行列数无关**。4×4 与 8×8 使用 **相同的 24 个 color 名称与规则**；仅 `dest` 表按 `(rows, cols)` 实例化。

支持的产品档位（专利）：**8 / 16 / 24** 色（本实现 `MAX_COLORS=24`，见 `color_usage.COLOR_PROFILES`）。

---

## 3. Color 的两类静态规则

### 3.1 方向总线（Directional Bus）

每个节点上 `dest[node][color]` 固定指向 **某一罗盘方向** 的相邻节点（若存在）：

```
east  → (r, c+1)    west  → (r, c-1)
south → (r+1, c)    north → (r-1, c)
```

**规则**：包注入时指定 `(src, dst, color)`；每 hop 沿该 color 的固定方向前进，直到 `current == dst` 停止。  
**与 dst 无关** — 符合专利 Dest 661「静态 next-hop，非运行时路由」。

### 3.2 维度序单播（Unicast XY / YX）

`dest` 表为空，由 `unicast_modes[color] ∈ {xy, yx}` 在 **每 hop** 根据包内 `dst` 选择下一跳（先 row 后 col 或反之）。

**用途**：不规则点对点、MoE all-to-all、spare 通道。  
**非** 集合通信的最优选择（hop 数多、易与 collectives 争用单 VN）。

---

## 4. 固定 Color 目录（24 色）

| ID | 名称 | 类型 | 方向/模式 | 主要用途 |
|----|------|------|-----------|----------|
| 0 | unicast_xy | unicast | xy | 单播 / systolic |
| 1 | unicast_yx | unicast | yx | 单播 / 与 0 正交并行 |
| 2 | bcast_row_east | bus | east | 广播：行内东向扩散 |
| 3 | bcast_col_south | bus | south | 广播：列内南向扩散 |
| 4 | gather_col_north | bus | north | 汇聚：列内北向 |
| 5 | gather_row_west | bus | west | 汇聚：行内西向 |
| 6 | reduce_col_north | bus | north | 归约：列内北向（独立 VN） |
| 7 | reduce_row_west | bus | west | 归约：行内西向 |
| 8–9 | allgather_row_east/west | bus | east/west | Allgather 行环 |
| 10–11 | allgather_col_south/north | bus | south/north | Allgather 列环 |
| 12–15 | allreduce_rs/ag_row/col | bus | 各方向 | Allreduce 四阶段 |
| 16–17 | bcast_row_west / bcast_col_north | bus | west/north | 根不在左上角时的广播 |
| 18–19 | gather_col_south / gather_row_east | bus | south/east | 根不在左上角时的汇聚 |
| 20–21 | reduce_col_south / reduce_row_east | bus | south/east | 根不在左上角时的归约 |
| 22–23 | spare_xy/yx | unicast | xy/yx | 预留 |

代码：`wsesim/network/color_catalog.py` → `CATALOG`, `build_ideal_plan(rows, cols)`。

---

## 5. 分配与复用规则

编译器 / `color_alloc.py` 遵循：

1. **时间重叠且链路相交** 的 flow → **必须不同 color**
2. **时间不重叠** 的 flow → **可复用** 同一 color
3. 目标：最小化 **峰值链路负载**（makespan）
4. 每 color 的转发图应 **无环**（或 credit 有界环）；`validate_plan()` 可检查

**保序（仿真）**：每个 `(color, src, dst)` 流串行注入；单 color 单缓冲 + 固定路由 ⇒ FIFO。

---

## 6. 各类集合通信如何使用 Color

以下描述 **理想静态路由**（`generate_ideal_collective`）。Baseline 为朴素直连 + 单 VN color 0，仅作对比。

### 6.1 Broadcast（广播）

**目标**：根 PE 的数据复制到 mesh 上所有 PE。

**Color**：`pick_broadcast_colors(root, rows, cols)`  
- 根在左/上：color **2**（行东）+ **3**（列南）  
- 根在右/下：color **16**（行西）+ **17**（列北）

**传输步骤**：

```
Phase 1 — 行扩散（root 所在行）
  根 → 沿 row_color 逐 hop 扩散到行内所有 PE
  delay = 0, 1, 2, …（流水线）

Phase 2 — 列扩散（每一列）
  根行上的 PE → 沿 col_color 向下/向上扩散到列内所有 PE
  delay = (cols-1) + row_offset
```

**示例（4×4，root=0）**：

| 步骤 | src→dst | color | delay |
|------|-----------|-------|-------|
| 行 0 东向 | 0→1, 1→2, 2→3 | 2 | 0,1,2 |
| 列 0 南向 | 0→4, 4→8, 8→12 | 3 | 1,2,3 |
| 列 1 南向 | 1→5, 5→9, … | 3 | 2,3,4 |
| … | … | 3 | … |

**makespan ≈ (cols−1) + (rows−1)**，全部相邻边。

---

### 6.2 Gather（汇聚）

**目标**：各 PE 的数据 **收集** 到根（不做归约运算，仅搬运）。

**Color**：`pick_tree_colors(root, gather=True)` → 通常 **4+5**（根在 (0,0)）

**传输步骤**（与 broadcast 相反）：

```
Phase 1 — 列汇聚
  每列最下行 → 沿 col_color 逐 hop 到根行

Phase 2 — 行汇聚
  根行上各 PE → 沿 row_color 逐 hop 到根
```

**与 reduce 的区别**：拓扑相同，但使用 **不同 color ID（4/5 vs 6/7）**，以便 gather 与 reduce 在融合 kernel 中 **时间重叠** 而不争用 VN。

---

### 6.3 Reduce（归约）

**目标**：各 PE partial sum **归约** 到根（CE 在每 hop 合并）。

**Color**：`pick_reduce_colors(root)` → **6+7**（根在 (0,0)）

**传输**：与 gather 相同的几何形状，独立 VN。  
每包 `payload="reduce"`，`size_bytes` 在仿真中不变（合并语义由 CE 建模，不在 NoC 层展开）。

---

### 6.4 Allgather（全收集）

**目标**：每个 PE 最终持有 **所有 PE** 的数据块（每 PE 贡献 1/N 块）。

**Color**：**8, 9, 10, 11** — 行东/西、列南/北四个环向 VN

**算法**：2D 环 allgather

```
Phase A — 行内 (cols−1 步)
  每步：每行内 east 环一步 + west 环一步（不同 color，链路不冲突）

Phase B — 列内 (rows−1 步)
  每步：每列内 south 环一步 + north 环一步
```

**每步每条边**：相邻 PE 互传一个 chunk；4 个 color 使 bidirectional 环 **并行**。

**makespan ≈ (cols−1) + (rows−1)**（与 broadcast 深度同阶，但语义为 ring reduce-scatter/allgather）。

---

### 6.5 Allreduce（全归约）

**目标**：每个 PE 得到 **全局归约结果**。

**Color**：**12, 13, 14, 15** — 四阶段独立 VN

```
Phase 1 — 行 reduce-scatter (cols−1 步)   color 12 (rs_row_east)
Phase 2 — 行 allgather     (cols−1 步)   color 13 (ag_row_west)
Phase 3 — 列 reduce-scatter (rows−1 步)  color 14 (rs_col_south)
Phase 4 — 列 allgather     (rows−1 步)   color 15 (ag_col_north)
```

各 phase 用 `delay_cycles` 时间错开（`t`, `t+cols-1`, …），同一时刻每个 phase 仅一个 color 活跃，避免环上冲突。

---

## 6.1 一次集合通信尽量少切 Color（ColorBudget）

编译器可为 **单次 collective 调用** 选择 color 预算策略（`ColorBudget`），在 **color 数量** 与 **并行度 / makespan** 之间权衡：

| 策略 | 每 collective 最多 distinct colors | 机制 |
|------|-----------------------------------|------|
| **MINIMAL** | **1** | 全程使用 `unicast_xy`（color 0）；行/列 phase、环上 east/west 通过 `delay_cycles` **时间串行**，同一 color 上不并发冲突流 |
| **COMPACT** | **2** | `unicast_xy` + `unicast_yx`：正向 hop（东/南）用 color 0，反向 hop（西/北）用 color 1；行 phase 与列 phase **复用同一对 color**（时间不重叠） |
| **PARALLEL** | 2（树）/ 4（环） | 当前默认：方向总线独立 VN，同一步 east/west 可并行 |

### 理论下界

- **方向总线** 每个 color 只编码 **一个罗盘方向**；2D 生成树需要 row + column 两种方向 → 纯总线方案 **至少 2 色**，无法再减。
- **环 / allreduce** 同一步需 east 与 west（或 south 与 north）→ 纯总线 **至少 2 色/维**；要压到 **1 色** 必须改用 **单播** 或 **时间串行**  opposing 方向。
- **非重叠时间** 可 **复用同一 color ID**（专利允许）；COMPACT 即利用 phase 边界复用 XY/YX 对。

### API

```python
from wsesim.network.color_usage import ColorBudget, distinct_colors, budget_for
from wsesim.network.collective import generate_ideal_collective

# 一次 broadcast 只用 1 个 color
traffic = generate_ideal_collective(
    "allgather", 8, 8, 128, color_budget=ColorBudget.MINIMAL
)
assert distinct_colors(traffic) == 1
assert budget_for("allreduce", ColorBudget.COMPACT) == 2
```

仿真对比三种预算 + baseline：

```python
from wsesim.network.color_sim import compare_all_budgets
results = compare_all_budgets(4, 4, "allreduce")
```

**选型建议**：kernel 内 color 切换有 wavelet task 开销时优先 **MINIMAL/COMPACT**；带宽敏感、环步需 bidirectional 并行时用 **PARALLEL**。

---

```python
from wsesim.network.color_catalog import build_ideal_plan
from wsesim.network.color_usage import (
    COLLECTIVE_SCENARIOS,
    ColorBudget,
    scenario_for,
    pick_broadcast_colors,
    validate_plan,
)
from wsesim.network.collective import generate_ideal_collective

# 构建 8×8 mesh 的 24 色 plan
plan = build_ideal_plan(8, 8)

# 生成 broadcast 理想流量（每包带 color 字段）；默认 PARALLEL
traffic = generate_ideal_collective("broadcast", rows=8, cols=8, chunk_bytes=128, root=0)

# 最少 color：整次 collective 仅 color 0
traffic_min = generate_ideal_collective(
    "allgather", 8, 8, 128, color_budget=ColorBudget.MINIMAL
)

# 查询场景规则
sc = scenario_for("allgather")
print(sc.colors, sc.route_shape)

# 校验 bus color 无环
assert validate_plan(plan) == {}
```

运行对比实验：

```bash
.venv/bin/python examples/run_color_vs_xy.py --msg-bytes 128
.venv/bin/python examples/generate_color_report_html.py
```

---

## 8. 与 Baseline 的对比要点

| 维度 | Baseline (`xy_single_vn`) | Color ideal |
|------|---------------------------|-------------|
| Color 数 | 1 | 24 |
| 路由 | 每包 XY 多跳至 dst | 相邻边 + 方向总线 |
| Broadcast | 根→每点直连 (N−1 条多跳流争用) | 生成树 O(rows+cols) 深度 |
| Allgather | 全连通 all-to-all | 2D 环，4 VN 并行 |
| 保序 | N/A | `(color,src,dst)` 锁，violations=0 |

---

## 9. 扩展场景（catalog 预留）

| 场景 | 建议 color | 说明 |
|------|------------|------|
| Systolic 脉动 | 0 (XY) / 1 (YX) | 行内邻居流；双 color striping |
| MoE all-to-all | 1 (YX) + 动态 alloc | 时间分相 + ILP 分配 |
| RHD allreduce | spare / 扩展至 32 色 | 每 XOR stage 一 color |
| 非 corner 根 | 16–19 | `pick_*_colors(root)` 自动选择 |
| Partial-good | `color_repair.py` | 缺陷图上重算 dest，保持形状 |

---

## 10. 参考

- 专利分析：`docs/color_mechanism_analysis.md`
- 仿真报告：`docs/color_simulation_report.html`
- 目录定义：`wsesim/network/color_catalog.py`
- 场景规则：`wsesim/network/color_usage.py` → `COLLECTIVE_SCENARIOS`
- 流量生成：`wsesim/network/collective.py` → `generate_ideal_collective`
- Cloud SDK（无 NoC）：`vendor/cerebras-cloud-sdk-python/README.md`

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

> ⚠️ **本目录是「软件设计选择」，不是硬件强制**。硬件只保证「每个 color = 一张任意静态多播路由表」；把「一个 color 绑一个罗盘方向」是一种实现，并非最优。下面 §4A 从专利重新推导后给出修正。

---

## 4A. 软硬协同与调度机制（专利 US10,515,303 重新推导）

> 本节回答核心问题：**Color 是 TDM 时分轮转，还是需要 trigger 切换的虚拟子网？** 并据此重推 color 划分规则。所有结论标注专利依据（Router 600 / Router Sched 654 / Picker 830 / FIGS. 6, 7A–7D, 8, 9A–9C）。

### A. 结论先行：既不是固定时隙 TDM，也不是"一次只激活一条子网"

Color 机制是 **两层** 的，两层都 **不是** 固定时隙 TDM 轮转：

| 层 | 机制 | 是否 TDM 轮转 | 是否 trigger |
|----|------|---------------|--------------|
| **路由层（Router）** | 多条 color 同时常驻、各有独立缓冲；共享物理链路时，对**就绪（ready）color** 做 **round-robin / priority 仲裁** | **否**：空闲 color 不占时隙，是 *按需* 仲裁（demand-driven），不是预分配固定 slot | 否：硬件自动仲裁 |
| **计算层（CE / Picker）** | wavelet 到达使该 color 的 `Active Bit` 置位；Picker 选中一个 *active 且未 block* 的 color → **触发** 对应 task（`addr = base + color×4`） | **否**：只在 active/unblock 的 color 间选，不是轮所有 color | **是**：color 是"激活→被选中→触发任务"的事件驱动开关 |

一句话：**链路上是"就绪色按需时分仲裁"，计算上是"事件触发激活 + Picker 选色起任务"**。没有任何"全局按固定顺序轮转所有 color"的时隙表，也没有"同一时刻全芯片只有一条 color 活着"的限制。

### B. 硬件依据（逐条出处）

1. **Color = 虚拟网络，独立缓冲、共享路由**
   > "An example fabric comprises 16 logically independent networks referred to as colors. Each color is a virtual network … Each color has dedicated physical buffering resources but shares the same physical routing resources … a fabric comprises various numbers of colors (e.g., 8, 24, or 32)." （Fabric Overview）

2. **静态固定路由，无运行时路由判断；多播靠静态多出口复制**
   > "Once configured … each color is a fixed routing pattern. All data that flows within a color always flows in accordance with the fixed routing pattern. **There are no dynamic routing decisions.**"
   > "To perform multicast, each router node is statically configured with **multiple outputs per multicast color**. The router replicates an incoming wavelet … to all outputs specified by the static configuration."
   → 仿真中即 `Dest 661`：`dest[node][color] → {方向集合}`，可多出口（多播）。

3. **每 color 单活跃输入源 → color 内无拥塞；拥塞只发生在 color 之间**
   > "The router provides for multiple input sources per color and **processes a single active input source at a time** … the router has a single buffer per color instead of a buffer per input source. Since there is only a single active input source at a time, there is not any congestion within a color. However … congestion occurs between colors since the colors share a single physical channel. **The router responds to the congestion by scheduling between ready colors onto a single shared output channel.**"

4. **链路仲裁策略 = round-robin 或 priority（这就是"时分"的真相）**
   > "each scheduler implements one or more scheduling policies, e.g., **round-robin and priority**. The round-robin … choosing between **all available colors** one at a time … The priority … choosing from among a first set of predetermined colors (e.g., colors 0-7) with higher priority than … a second set (e.g., colors 8-15)."
   → 关键词是 *available / ready colors*：只有有数据且下游未 stall 的 color 才参与仲裁（`Sent 662`：未发过且方向未 stall 才挑）。**空闲 color 不消耗带宽**，故非固定时隙 TDM，而是统计复用。

5. **每 color 独立反压（credit/stall），沿路由反向传播**
   > "There is an **independent backpressure channel for each color** … When a color is back pressured, data queued at each hop within the fabric is stalled … the queued data is an extension to a queue at the destination."
   → 仿真中 `Stall Out 630 / Stall In 640`，每 color × 每方向一个 stall 位；缓冲默认 **2 entries/color**（专利 `Data Queues 650 = colors × 27/33 bit × 2 entries`）。

6. **Color → Task：选中 color 即触发计算（trigger 语义来源）**
   > 数据 wavelet：`Add (Color×4) to Base Register to Form Instruction Address`；控制 wavelet：用 index 低 6 位。Picker "selects an **active unblocked color** for processing to **initiate a corresponding task**"。

7. **显式激活 / 阻塞（软件对 trigger 的精确控制）**
   - `Active Bit 898`：wavelet 写入 Input Q 时自动置位；也可由 **activate 指令**（立即数/寄存器指定 color）或 **DSD 的 AC 字段**（处理完 fabric 向量后激活某 color）或 **环形缓冲 push/pop color** 激活。（FIG. 9B）
   - `Block Bit 899`：`block/unblock` 指令对某 color 门控——被 block 的 color，其 wavelet 不被选、其 active 状态也不触发 task，直到 `unblock`。（FIG. 9C）
   → 因此软件能精确编排"何时让某条 color 的任务可被触发"，这是 trigger 模型而非时隙模型。

8. **优先级类**：低 ID color（0–7）可被设为高优先级（patent EC61/EC63 + priority policy），适合 control/closeout/延迟敏感流。

### C. 软硬协同完整流程（编译期 → 配置期 → 运行期）

```
[编译期] Placement Server SW / Neuron-to-PE Mapping SW（FIG.2）
  ├─ 决定 color 总数档位（8/16/24/32）
  ├─ 为每条 color 生成静态 Dest 表（含多播多出口），保证无依赖环
  ├─ 把 color → task 起始地址（base + color×4）写入指令表
  └─ 规划每 color 的 Input Q 容量 / 优先级类

        │ 配置经"预定 color（如 color 0）+ 固定多播"分发
        ▼
[配置期] Connection Server SW / Misc SW on FPGAs（经 color 0 广播路由与 CE 配置）
        │
        ▼
[运行期] 每个 PE 内（FIG.6/8）：
  Router:  Data In → Write Dec → per-color Data Queue(2 entries)
           → Router Sched（RR/priority 在 ready colors 间仲裁）→ Data Out
           ⇅ per-color 反压 Stall In/Out
  CE:      Off Ramp → Hash 选 Input Q（按 color）→ 置 Active Bit
           → Picker（RR / pick-from-last，选 active & !block 的 color）
           → addr=base+color×4 → 取指执行 → terminate → 选下一个 color
           → 输出经 Output Queues + On Ramp 回 Router
```

软硬分工本质：**软件负责"空间"（编译期把通信图映射成静态多播路由 + color 划分），硬件负责"时间"（运行期按就绪/优先级在 color 间仲裁链路、按激活/阻塞触发任务）**。软件不在运行时切路由，只通过 activate/block/反压 影响"哪些 color 此刻就绪"。

### D. 从性能 / 容错 / 工作负载重新推导划分规则

由 §B 的 8 条硬件事实，导出 5 条 **第一性原理**，并指出现有 catalog 的可改进点：

| # | 硬件约束 | 推导出的划分规则 |
|---|----------|------------------|
| R1 | 每 color 一张静态多播表，可多出口 | **一棵多播树 = 1 个 color**。2D broadcast 让每节点同时向 east+south 复制 → **整个广播只需 1 色**，无需"行 1 色 + 列 1 色"。 |
| R2 | 每 color 单活跃输入源 | 扇入型（reduce/gather）同一节点多个子节点共用一色时须 **时间串行**（或 CE 合并）；只要每节点同一时刻一个源即合法。 |
| R3 | color 内无环才不死锁（无硬件死锁规避） | 双向环 / allreduce 必须 **拆成无环链**：opposing 方向放不同 color，或用相位 + delay 打断环。**拆 2 色/维 的真正理由是无环，不是带宽。** |
| R4 | 共享链路对 ready color 做 RR | 同一条物理链路上铺 K 条 active color，每条≈1/K 带宽 → **多 color 不增加总带宽**；只有走 **不相交链路** 才真正并行加速。 |
| R5 | color 总数稀缺 + 每色有 task/Q/Active/Block 成本；低 ID 可高优先级 | **最少化单次 collective 的 distinct color**（见 §6.1 ColorBudget）；control/closeout/延迟敏感放 color 0–7 高优先级。 |

**据此修正现有集合通信规则（标注"硬件最优" vs "软件惯例"）：**

| 集合通信 | 现有 catalog 规则 | 重推后的硬件最优 | 说明 |
|----------|------------------|------------------|------|
| **Broadcast** | 2 色（行东 2 + 列南 3） | **1 个多播色**（每节点静态 east+south 双出口） | R1。现规则把树拆成两维两色属冗余；单色即可 O(rows+cols) 完成且零 task 切换 |
| **Reduce / Gather** | reduce 6/7、gather 4/5 各 2 色 | **1 个扇入色**即可（单活跃源串行 / CE 合并） | R2。分 4/5 vs 6/7 **仅在** gather 与 reduce 在融合 kernel 内 **时间重叠** 才需要 |
| **Allgather（双向环）** | 4 色（行东/西 + 列南/北） | **2 色/相位**：同一相位只跑一个方向链，反向相位 delay 错开 | R3+R4。共享链路上 east/west 同时跑并不加带宽，4 色主要价值是"非阻塞"，可用相位换 color |
| **Allreduce** | 4 色（RS/AG × 行/列） | **2 色复用**（XY/YX）跨 4 相位时间复用，或 4 色换并行度 | R4+R5。即 `ColorBudget.COMPACT`；4 色仅在各相位走不相交链路时才有并行收益 |

> 结论：现有 24 色 catalog 偏向「方向总线 + 一通信一专用色」的 **可读性/并行优先** 设计；从 **color 稀缺性 + RR 带宽不叠加** 看，多数 collective 可压到 1–2 色（broadcast 甚至 1 个多播色），这正是 §6.1 `ColorBudget` MINIMAL/COMPACT 的硬件依据。选择哪种取决于：是否带宽受限（走不相交链路才用多色）、是否在意 task 切换开销、color 预算是否紧张。

### E. 容错与工作负载维度

- **容错**：路由全静态 + 看门狗（"watchdog mechanism detects lack of progress and signals a fault"）。出现坏 PE/链路时，**离线重算每 color 的 Dest 表**绕过缺陷点（`color_repair.py`），重算须保持 **R3 无环 + R2 单源**；建议保留 spare 色（22/23）做冗余通道。
- **工作负载（专利钦定的轴映射）**：
  > 软件用 **水平维** 做 *层间* 通信（activation 广播），用 **垂直维** 做 *层内* 通信（partial-sum 累加，常为 ring）。
  → 因此 color 划分应按 **轴 + mega-phase（forward/delta/chain）** 归组：水平色族给 broadcast/activation，垂直色族给 reduce/partial-sum；三个 mega-phase 共用同一 PE 数据通路，须靠 R3「至少一个任务保证完成」打破跨相位环。

### F. 与 `cerebras-cloud-sdk-python` 的边界（再确认）

`vendor/.../resources/` 仅有 `chat`、`completions`、`models` 三个 OpenAI 风格 REST 资源；全仓 `grep` **零命中** `wavelet/color/fabric/router/allreduce/noc`。即：**Cloud SDK 完全不暴露 color/fabric**——color 是 *片上编译期* 概念，软硬协同发生在 Placement/Connection Server SW + 片上 Router/CE，而非云端 SDK。本仓 `wsesim` 是对该片上机制的 cycle-accurate 建模。

---

## 5. 分配与复用规则

编译器 / `color_alloc.py` 遵循（依据见 §4A.D 的 R1–R5）：

1. **时间重叠且链路相交** 的 flow → **必须不同 color**（R4：同链路 RR 不叠带宽，仅为非阻塞）
2. **时间不重叠** 的 flow → **可复用** 同一 color（专利允许；COMPACT/MINIMAL 的基础）
3. 目标：最小化 **峰值链路负载（makespan）** 与 **distinct color 数**（R5：color 稀缺，每色含 task/Q/Active/Block 成本）
4. 每 color 的转发图必须 **无环**（硬件无死锁规避，靠软件保证；R3），`validate_plan()` 可检查
5. **一棵多播树 = 1 色**（R1）：能用静态多出口复制覆盖的扇出，不要拆成多色
6. control/closeout/延迟敏感流 → color **0–7**（高优先级类）

**保序（仿真）**：每个 `(color, src, dst)` 流串行注入；单 color 单缓冲 + 固定路由 ⇒ FIFO。

---

## 6. 各类集合通信如何使用 Color

以下描述 **理想静态路由**（`generate_ideal_collective`）。Baseline 为朴素直连 + 单 VN color 0，仅作对比。

### 6.1 Broadcast（广播）

**目标**：根 PE 的数据复制到 mesh 上所有 PE。

**Color**：`pick_broadcast_colors(root, rows, cols)`  
- 根在左/上：color **2**（行东）+ **3**（列南）  
- 根在右/下：color **16**（行西）+ **17**（列北）

> 📌 **硬件最优修正（§4A.D R1）**：因每节点可对同一 color 静态配置 **多出口**，整棵 2D 广播树（每节点同时向 east+south 复制）可仅用 **1 个多播色** 完成，无需行/列两色。现行 2 色方案是「方向总线可读性」惯例，非硬件下界；带宽不受限时优先 1 多播色（亦即 `ColorBudget.MINIMAL` 的硬件依据）。

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

**交互可视化**：[`docs/color_mesh_viz.html`](color_mesh_viz.html) — 逐 cycle 展示 Color VN 上的 flit 传输、VN 泳道、预算对比（打开即用，无需服务器）。与 [`docs/collective-mesh-viz.html`](collective-mesh-viz.html) 的逻辑算法视图互补。

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

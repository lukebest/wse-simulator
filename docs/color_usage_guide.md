# Color 用法与规则指南

本文档说明 Cerebras WSE 上 **Color（虚拟网络）** 的语义、规则，以及在各类集合通信场景下如何用 color 传输数据。实现代码位于 `wsesim/network/color*.py`；硬件依据为专利 **US10,515,303**。

---

## 1. 与 Cerebras SDK 2.10.0 的关系

| 层次 | 路径 / 产品 | 是否包含 Color |
|------|-------------|----------------|
| **Cloud REST API** | PyPI `cerebras-cloud-sdk` · [GitHub](https://github.com/Cerebras/cerebras-cloud-sdk-python) | **否** — HTTP 客户端（chat completions、models） |
| **片上 SDK / CSL** | [Cerebras SDK 2.10.0](https://sdk.cerebras.net/) · 本仓库 `docs/vendor/cerebras-sdk-2.10.0/` | **是** — `@get_color`、`@set_color_config`、wavelet-triggered tasks |
| **NoC 仿真** | `wsesim/network/` + 本文档 | **是** — cycle-accurate 建模（专利 + catalog） |

Cloud SDK 与 Wafer SDK 是不同产品：前者是云端推理 REST，后者在集群上用 CSL 开发 kernel 并配置 fabric color。本仓库 **不 vendoring** 任一 SDK 源码；片上 color 语义见 `docs/vendor/cerebras-sdk-2.10.0/fabric-and-color.md` 与专利 US10,515,303。

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

## 4A. 软硬协同与调度机制（专利 US10,515,303 + SDK 2.10.0 重新推导）

> 本节回答核心问题：**Color 是 TDM 时分轮转，还是需要 trigger 切换的虚拟子网？** 并据此重推 color 划分规则。结论由 **硬件**（专利 Router 600 / Router Sched 654 / Picker 830 / FIGS. 6, 7A–7D, 8, 9A–9C）与 **软件**（Cerebras SDK 2.10.0 CSL builtins：`@set_color_config`、`@get_data_task_id`、`@block`/`@unblock`、switches）**双向印证**。SDK 摘要见 `docs/vendor/cerebras-sdk-2.10.0/`。

### A. 结论先行：不是 TDM；是"常驻多色 + 三种触发"

先回答二选一：**都不是简单的那一种**。准确表述是——

- **不是 TDM**：没有"为每条 color 预分配固定时隙、按全局顺序轮转"的时隙表。空闲 color 不占带宽。
- **不是"一次只激活一条子网"**：所有 color 的路由表 **同时常驻** 每个 Router，多色可同时在飞。
- **确实存在 trigger**，而且是 **三个不同层面** 的触发，必须分开看：

| 层 | 机制 | TDM？ | trigger？ | 硬件依据 | 软件依据(SDK 2.10.0) |
|----|------|-------|-----------|----------|----------------------|
| **L1 链路仲裁（Router）** | 共享物理链路时，对 **ready** color 做 round-robin / priority 仲裁 | **否**（按需统计复用，非固定 slot） | 否（硬件自动） | Router Sched 654；"scheduling between ready colors" | 透明，CSL 不可见 |
| **L2 任务触发（CE / Picker）** | wavelet 到某 color → `Active Bit` 置位 → Picker 选 *active 且 unblock* 的 color → 起 task（`addr=base+color×4`） | **否**（只在 active/unblock 间选） | **是**：事件触发计算（WTT） | Picker 830；FIG 9A–9C | "A data task can be scheduled **iff** it is bound to a data task ID that is **activated and unblocked**"；`@bind_data_task` |
| **L3 路由切换（Router 配置）** | **route-switching control wavelet** 让某 color 的路由表在 **预编译的 pos0→pos1→pos2→pos3** 间前进（`ring_mode` 可回绕） | 否（按 control wavelet 推进，非时钟轮转） | **是**：trigger 切换该 color 的 VN 拓扑 | 控制 wavelet 配置路由（配置期同机制） | `@set_color_config` 的 `switches.pos1/2/3` + `ring_mode` |

**对用户问题的精确回答**：

1. "TDM 轮转时分？" → **否**。链路上是 *就绪色按需仲裁*（L1），不是固定时隙。
2. "需要 trigger 切换成某条 color 虚拟子网？" → **部分是，但要分清切什么**：
   - 计算侧（L2）：wavelet 是"激活并触发某 color 任务"的 trigger——但它 **不切换** color，多色可并存。
   - 路由侧（L3）：**存在** 真正"用 trigger 切换虚拟子网"的机制——`route-switching control wavelet` 让 **同一个 color ID** 在 ≤4 套 **编译期预设** 的路由拓扑间前进。这才是字面意义的"trigger 切换 VN"，但切的是 **一条 color 内部的路由相位**，不是"在不同 color 间切换"。

一句话：**链路按就绪统计复用（L1，非 TDM）；计算由 wavelet 事件触发任务（L2）；单条 color 的路由可由控制 wavelet 在预设相位间被 trigger 切换（L3，ring_mode 回绕）**。没有全局时隙表，也没有"全芯片同时只有一条 color"的限制。

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

### B′. 软件依据（Cerebras SDK 2.10.0，与硬件逐条对应）

SDK 是程序员 **配置面**，与上面硬件 **机制面** 一一印证（详见 `docs/vendor/cerebras-sdk-2.10.0/fabric-and-color.md`）：

| # | SDK 事实（CSL builtin / 文档原文） | 对应硬件结论 |
|---|-----------------------------------|--------------|
| S1 | "IDs **0 to 23** … recognized by the hardware as **virtual communication channels**"，`@get_color(n)` | = §B1 color 是 VN（catalog 0–23） |
| S2 | `@set_color_config(x,y,c,.{.routes=.{.rx=…,.tx=…}})`：编译期为每 PE 每 color 配静态 rx/tx；`tx` 可多方向 | = §B2 静态路由 + **多出口多播**（`.tx=.{RAMP,EAST}` 即 Dest 661 多 bit） |
| S3 | "only safe to enable **multiple input directions** … if wavelets will **never arrive from multiple directions at once**; otherwise router behavior is **undefined**" | = §B3 **每 color 单活跃输入源**（这是 R2 的软件铁证） |
| S4 | data task "can be scheduled **iff** … **activated and unblocked**"；"activated by **receiving a wavelet** along a given color" | = §B6 color→task 触发（WTT，L2） |
| S5 | `@activate` / `@block` / `@unblock`（输入为 color 或 data_task_id） | = §B7 Active/Block 位的软件接口 |
| S6 | `switches.pos1/pos2/pos3` + `ring_mode`：route-switching control wavelet 推进路由相位（详见 B″） | = L3 路由切换（**新增维度**，专利由控制 wavelet 配置路由同源） |
| S7 | `filter`（counter / sparse_counter / range）：按计数/索引筛选哪些 wavelet 进 CE | 多 PE 共享一条 color、各取子集（节省 color，见 R6） |
| S8 | WSE-2：data_task_id 由 **color** 构造；WSE-3：由 **input_queue** 构造，`@initialize_queue(iq,.{.color=c})` 显式绑定；switch 仅 color {0–9,12,13,16,17,20,21} 支持 | color↔task 解耦演进；L3 在 WSE-3 上 **按 color 受限** |

> 软件 **不能** 在运行时任意改路由：`@set_color_config` 只在 layout/comptime（编译期）调用，同一 (PE,color) 只能配一次。运行时的"路由变化"**只有 L3**——在 **编译期已写好的 pos0–pos3** 之间，由 control wavelet 触发前进，外加 teardown 模式下经标准库重配。**这与"无动态路由判断"不矛盾**：候选拓扑全部静态预置，trigger 只选相位。

### B″. L3 路由切换机制详解（"trigger 切换 VN"的真身）

这是回答"需要 trigger 切换虚拟子网吗"的关键，SDK 原文（`@set_color_config` Switching Semantics）：

> "A route configuration for a given color can **change dynamically through control wavelets** … `pos1`,`pos2` and `pos3` are additional configurations we can **switch to in-sequence** … the first route-switching control wavelet will set `pos1` … the third will cause an advance to `pos3`. … if `ring_mode` … loop-back to the original configuration once all valid switch positions have been visited."

要点：

- **切的是"一条 color 的路由相位"**，不是 color 之间。一个 color ID 可携带最多 **4 套**（pos0 初始 + pos1/2/3）编译期预设的 rx/tx 拓扑。
- **触发源是 control wavelet**（与配置期下发路由表同一类机制），不是时钟、不是数据 wavelet。
- `pop_mode`（no_pop / always_pop / pop_on_advance[_nop]）控制 control wavelet 携带的指令序列如何随经过 PE 而消耗。
- `ring_mode=true` → 相位走完回到初始，天然适合 **周期性多相集合通信**（如 allreduce 的 RS/AG 交替、双向环的方向交替）。
- **WSE-3 限制**：只有 color {0–9,12,13,16,17,20,21} 支持 switch；其余 color 只能单一静态拓扑。

→ 这直接给出第 6 条划分原理（见 §D 的 R6）：**用 switch 相位在一条 color 内时分多套拓扑，可把"多相/双向"集合通信压到更少的 distinct color**，而不必每相位/每方向各占一个 color ID。

### C. 软硬协同完整流程（编译期 → 配置期 → 运行期）

```
[编译期] CSL 程序 + SdkLayout（@set_tile_code / @set_color_config / @get_color）
  ├─ 决定 color 总数档位（8/16/24/32）与 0–23 的应用语义
  ├─ 为每条 color 生成静态 rx/tx 路由（tx 多方向=多播多出口），保证无依赖环
  ├─ 绑定 color→task：@bind_data_task（WSE-2 由 color；WSE-3 由 input_queue）
  ├─ 预置 switches（pos1/2/3 + ring_mode）、filter、优先级类、Input Q 容量
  └─ cslc 编译 → 二进制 + 每 PE 配置流

        │ 配置经"预定 color + 固定多播"下发（控制 wavelet 写路由/配置寄存器）
        ▼
[配置期] SdkRuntime / appliance：加载程序、初始化 queue、@unblock 起始 color
        │
        ▼
[运行期] 每个 PE 内（FIG.6/8）：
  Router:  Data In → Write Dec → per-color Data Queue(2 entries)
           → Router Sched（RR/priority 在 ready colors 间仲裁）→ Data Out
           ⇅ per-color 反压 Stall In/Out
           ↺ L3：route-switching control wavelet → 某 color 路由 pos0→pos3（ring 回绕）
  CE:      Off Ramp → Hash 选 Input Q（按 color）→ 置 Active Bit
           → Picker（RR / pick-from-last，选 active & !block 的 color）
           → addr=base+color×4 → 取指执行 → terminate → 选下一个 color
           → 输出经 Output Queues + On Ramp 回 Router
```

软硬分工本质：**软件负责"空间 + 预案"（编译期把通信图映射成静态多播路由、color 划分、以及每 color 的 ≤4 套路由相位预案），硬件负责"时间"（运行期按就绪/优先级在 color 间仲裁链路、按 wavelet 激活/阻塞触发任务、按 control wavelet 在预案相位间推进路由）**。关键澄清：运行期 **唯一** 的路由变化是 L3 在 **编译期预置的 pos0–pos3** 间被 trigger 切换，仍属"无动态路由计算"——没有按目的地实时算路由。

### D. 从性能 / 容错 / 工作负载重新推导划分规则

由 §B 的 8 条硬件事实 + §B′ 的 SDK 印证，导出 6 条 **第一性原理**（R1–R6），并指出现有 catalog 的可改进点：

| # | 硬件约束 | 推导出的划分规则 |
|---|----------|------------------|
| R1 | 每 color 一张静态多播表，可多出口 | **一棵多播树 = 1 个 color**。2D broadcast 让每节点同时向 east+south 复制 → **整个广播只需 1 色**，无需"行 1 色 + 列 1 色"。 |
| R2 | 每 color 单活跃输入源 | 扇入型（reduce/gather）同一节点多个子节点共用一色时须 **时间串行**（或 CE 合并）；只要每节点同一时刻一个源即合法。 |
| R3 | color 内无环才不死锁（无硬件死锁规避） | 双向环 / allreduce 必须 **拆成无环链**：opposing 方向放不同 color，或用相位 + delay 打断环。**拆 2 色/维 的真正理由是无环，不是带宽。** |
| R4 | 共享链路对 ready color 做 RR | 同一条物理链路上铺 K 条 active color，每条≈1/K 带宽 → **多 color 不增加总带宽**；只有走 **不相交链路** 才真正并行加速。 |
| R5 | color 总数稀缺 + 每色有 task/Q/Active/Block 成本；低 ID 可高优先级 | **最少化单次 collective 的 distinct color**（见 §6.1 ColorBudget）；control/closeout/延迟敏感放 color 0–7 高优先级。 |
| R6 | 一条 color 可预置 ≤4 套路由相位（switches + `ring_mode`），由 control wavelet 触发推进（SDK S6/B″） | **多相 / 双向 集合通信可用 switch 相位时分复用 1 条 color**：allreduce 的 RS/AG、双向环的 east↔west 作为 pos0↔pos1↔…，`ring_mode` 回绕周期复用，免去每相位/每方向各占一个 color ID。WSE-3 限 color {0–9,12,13,16,17,20,21}。 |

**据此修正现有集合通信规则（标注"硬件最优" vs "软件惯例"，并标注 SDK 依据）：**

| 集合通信 | 现有 catalog 规则 | 重推后的硬件最优 | 说明（依据） |
|----------|------------------|------------------|------|
| **Broadcast** | 2 色（行东 2 + 列南 3） | **1 个多播色**（每节点静态 `.tx=.{EAST,SOUTH}` 双出口） | R1+S2。现规则把树拆成两维两色属冗余；单色即可 O(rows+cols) 完成且零 task 切换 |
| **Reduce / Gather** | reduce 6/7、gather 4/5 各 2 色 | **1 个扇入色**即可（单活跃源串行 / CE 合并） | R2+S3。分 4/5 vs 6/7 **仅在** gather 与 reduce 在融合 kernel 内 **时间重叠** 才需要 |
| **Allgather（双向环）** | 4 色（行东/西 + 列南/北） | **1–2 色**：用 switch 相位（pos0=east、pos1=west，`ring_mode`）在单色内交替方向；或 2 色非阻塞 | R3+R4+**R6**。共享链路上 east/west 同时跑并不加带宽；方向交替用 switch 相位替代独立 color |
| **Allreduce** | 4 色（RS/AG × 行/列） | **1–2 色**：4 相（RS行/AG行/RS列/AG列）映射到 pos0–pos3，control wavelet 逐相推进 | R4+R5+**R6**。即 `ColorBudget` 思路的硬件上界；4 色仅在各相位走不相交链路且要并行时才用 |

> 结论：现有 24 色 catalog 偏向「方向总线 + 一通信一专用色」的 **可读性/并行优先** 设计。从硬件第一性看有 **两条** 压缩路径：(i) **空间** 上一棵多播树=1 色、扇入=1 色（R1/R2）；(ii) **时间** 上多相/双向用 switch 相位在单色内时分（R6）。两者叠加后，broadcast 可 1 色、allreduce/allgather 也能 1–2 色，远少于现 catalog。是否真要多色，取决于：是否带宽受限（**只有走不相交链路** 多色才加速，R4）、是否在意 task/相位切换开销、color 预算与 WSE-3 的 switch 受限集。

### E. 容错与工作负载维度

**容错（性能-鲁棒权衡）**：

- 路由全静态 + 看门狗（"watchdog mechanism detects lack of progress and signals a fault"）。出现坏 PE/链路时 **离线重算每 color 的路由**绕过缺陷点（`color_repair.py`），重算须保持 **R3 无环 + R2 单源**。
- **SDK 工具**：`teardown` 模式可让某 color 启动即挂起、运行时经标准库 **重配路由/filter**——为"局部故障后不重编译、在线改道"提供软件钩子；`filter`（counter/range）可在共享 color 上 **丢弃/隔离** 异常索引的 wavelet。
- 建议保留 spare 色（22/23）做冗余通道；**WSE-3 上若依赖 L3 改道，故障 color 必须落在 switch 支持集 {0–9,12,13,16,17,20,21}**，否则无法用相位改道，只能整表重算。

**工作负载（专利钦定的轴映射）**：

> 软件用 **水平维** 做 *层间* 通信（activation 广播），用 **垂直维** 做 *层内* 通信（partial-sum 累加，常为 ring）。

→ color 划分按 **轴 + mega-phase（forward/delta/chain）** 归组：水平色族给 broadcast/activation，垂直色族给 reduce/partial-sum。三个 mega-phase 共用同一 PE 数据通路，须靠 R3「至少一个任务保证完成」打破跨相位环；**周期性相位（如 ring/allreduce 的多步）优先用 R6 的 switch 相位在单色内推进，把稀缺的 color ID 留给真正需要不相交链路并行的流**。

### F. 与 Cerebras SDK / Cloud SDK 的边界（再确认）

| 产品 | 暴露 color/fabric？ | 在本分析中的角色 |
|------|---------------------|------------------|
| **Cerebras SDK 2.10.0**（CSL：`@get_color`/`@set_color_config`/`@bind_data_task`/switches） | **是** — 编译期静态路由 + WTT + L3 相位切换 | 软件配置面，印证 L1–L3（§B′/B″） |
| **cerebras-cloud-sdk**（REST chat/completions/models） | **否** | 与片上 color 无关 |

详见 `docs/vendor/cerebras-sdk-2.10.0/`（README / fabric-and-color / release-notes-2.10.0）。本仓 `wsesim` 结合 SDK 文档与专利 US10,515,303 做 cycle-accurate 建模；当前实现以 catalog + 方向总线为主，**L3 switch 相位复用与 WSE-3 queue↔color 解耦为后续可扩展项**。

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
- Cerebras SDK 2.10.0 参考：`docs/vendor/cerebras-sdk-2.10.0/`
- Cloud REST SDK（无 NoC）：https://github.com/Cerebras/cerebras-cloud-sdk-python

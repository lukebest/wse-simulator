# 完美时隙中继日历（Perfect Slot Relay Calendar）

研究问题：在 2D Mesh + 固定 XY 维序路由、每条 (src,dst) 流分配独立 color（时分 VN）的前提下，
能否构造一张**周期性（稳态）全局日历**，使得任意 router 在某全局时隙 `t` 入口收到的数据，
都能在**下一个全局时隙 `t+1`** 从某出口端口发出，且全网无冲突。

把「ingress slot t → egress slot t+1、每跳恰好 +1、全局无冲突」定义为 **time_slot_tolerance = 1 的完美解**。
若该解存在则 T=1 可行；否则放宽到 T>1（每端口最多 `k=T-1` 时隙缓冲，等价于全局日历中允许的偏移）。

本文给出形式化模型、下界、可判定性，并用 `docs/color_mesh_viz.html` 内置求解器在 4×4 / 8×8 上给出实测结论。

---

## 1. 形式化模型（周期 mod P）

设帧长（周期）为 `P`，日历无限重复。每条流 `f` 的 XY 路径为有向边序列
`P_f = (e_0, e_1, …, e_{L_f-1})`，并分配一个起始相位 `phi_f ∈ Z_P`。

- **每跳推进量** `step ∈ [1, T]`。T=1 时每跳恰好 +1，路径上的占用是一条**刚性对角线**：
  边 `e_h` 在时隙 `(phi_f + h) mod P` 被占用。
- **无冲突（周期意义）**：任意两条流 `f, g` 若共享同一条**有向边** `e`（在各自路径中的跳序为 `h_f, h_g`），
  则要求

  ```
  phi_f + h_f  ≢  phi_g + h_g   (mod P)
  ```

  即同一有向链路、同一时隙（mod P）至多一个 flit。
- **T=1 完美解** = 存在一组相位 `{phi_f}` 满足上述全部约束（每跳 +1、相位沿路径连续、全局无冲突）。
- **放宽 T>1** = 每跳可推进 `[1,T]`，等价于在中间 router 至多缓冲 `T-1` 拍；
  边 `e_h` 的时戳变为 `phi_f + Σ_{j≤h} step_j (mod P)`，给出额外松弛以消解冲突。

```mermaid
flowchart LR
  inj["inject @ phi_f"] -->|"+1"| h0["edge e0 @ phi_f"]
  h0 -->|"+1 (T=1) / +1..T"| h1["edge e1 @ phi_f+1"]
  h1 -->|"..."| hk["edge e_h @ phi_f+h (+dwell)"]
  hk --> ok["all mod P; no two flows share (edge, slot)"]
```

> 稳态约定：周期日历分析中，每条 (src,dst) 流在一帧内只发**一个 flit**（一条对角线）。
> 数据量（多 flit）由帧的重复承载，不改变路由/争用结构——这与「color calendar 每帧给每个 color 一个时隙」的语义一致。

---

## 2. 下界：周期 P ≥ 最大有向边负载 L*

定义 `L* = max_e (经过有向边 e 的流数)`。每条流每帧在 `e` 上占用一个时隙，
而 `e` 每帧只有 `P` 个时隙，故任何无冲突周期日历必满足

```
P ≥ L*
```

`L*` 同时给出**不可行见证**：当 T=1 在 `P=L*` 不可行时，最拥塞的有向边即瓶颈所在。

---

## 3. 可判定性与求解

- **T=1**：对角线刚性，仅相位 `phi_f` 为自由变量。对相位做回溯搜索（按「最拥塞边归属 / 路径长」降序定序，
  贪心取最小可行相位，冲突则回溯）在迭代预算内是**完备判定**：找到即可行，搜尽预算未果记为 inconclusive。
- **T>1**：在每个相位下对每跳贪心取 `[1,T]` 内最小可行 step（启发式）。找到即可行（合法上界）；
  未找到不严格等价于不可行。

实现见 `docs/color_mesh_viz.html`：

- `computeEdgeLoad(flows)` → `{L*, witnessEdge}`
- `buildPeriodicCalendar(flows, P, T)` → `{ok, assign(phi,slots), exhausted, witness}`
- `findMinTolerance(flows, P=L*)`、`findMinPeriod(flows)`（T=1）
- `verifyPeriodic(flows, P, assign)` → 独立复核 `(有向边, 时隙 mod P)` 无重复
- `buildPeriodicSchedule(...)` → 生成可视化用的周期日历（列 `0..P-1`）

---

## 4. 各 collective 的结构与结论（实测）

求解器在 4×4 / 8×8、`P=L*`、T=1 下的结果（`verifyPeriodic` 全部独立复核通过，
所有 T=1 解满足每跳 +1 / dwell=0）：

| Collective | flows (4×4) | L* (4×4) | T=1 @ P=L* | minP @ T=1 | flows (8×8) | L* (8×8) | T=1 @ P=L* | minP @ T=1 |
|------------|-------------|----------|------------|------------|-------------|----------|------------|------------|
| broadcast  | 5   | 1  | 可行 | 1  | 9   | 1  | 可行 | 1  |
| reduce     | 15  | 1  | 可行 | 1  | 63  | 1  | 可行 | 1  |
| allreduce  | 64  | 2  | 可行 | 2  | 384 | 5  | 可行 | 5  |
| allgather  | 64  | 2  | 可行 | 2  | 384 | 5  | 可行 | 5  |
| gather     | 15  | 12 | 可行 | 12 | 63  | 56 | 可行 | 56 |

**核心结论：上述全部 collective 在 T=1 下均可行，且达到理论最优周期 `P = L*`。
`minT@P=L* = 1` 对所有情形成立——这些结构化流量不需要放宽到 T>1。**

### 4.1 broadcast / reduce —— 边不相交，T=1 平凡可行
`L*=1`：每条有向边只被一条流经过，故**不存在共享边 → 无任何约束**，任意相位都无冲突。
周期下界 `P=1`（退化帧：每条边每帧占用一次；相位差异不再必要）。这是多播树 / 2D 归约树的边不相交性所致。

### 4.2 allreduce / allgather —— 维度序 RHD，小幅共享，T=1 可达 P=L*
dim-wise RHD 的同一 stride 阶段内，配对双向流会共享少量链路，`L*` 较小（4×4 为 2，8×8 为 5）。
求解器用相位交替即可无冲突。例（allreduce 4×4，`P=2`，witness 边 `(0,0)->(1,0)`）：

```
ar_x1_ab_0_0  phi=1  slots=[1]  path=(0,0)->(1,0)
ar_x1_ba_0_0  phi=0  slots=[0]  path=(1,0)->(0,0)
ar_x1_ab_2_0  phi=0  slots=[0]  path=(2,0)->(3,0)
ar_x1_ba_2_0  phi=1  slots=[1]  path=(3,0)->(2,0)
```

### 4.3 gather —— root 热点决定周期，但 T=1 仍达最优
gather 让每个非 root PE 沿 XY 汇聚到 root，**进入 root 的边**承载最多流：
`L* = 12`（4×4）/ `56`（8×8），witness 边即 `(1,0)->(0,0)`（root 东邻入边）。
周期被热点链路下界 `P=L*` 限定，但 T=1 仍可行——因为汇聚到同一条边的多条流在该边上的**跳序不同**，
天然错开时隙。例（gather 4×4，`P=12`，相位均取 0，靠 hop 索引错开）：

```
g1  phi=0  slots=[0]      path=(1,0)->(0,0)              # 在 root 入边的 hop0 → slot 0
g2  phi=0  slots=[0,1]    path=(2,0)->(1,0)->(0,0)        # root 入边的 hop1 → slot 1
g3  phi=0  slots=[0,1,2]  path=(3,0)->(2,0)->(1,0)->(0,0) # root 入边的 hop2 → slot 2
g4  phi=0  slots=[0]      path=(0,1)->(0,0)              # 来自南邻的另一入边
```

注意：gather 是延迟/吞吐受限于 root 端口带宽的固有热点，`P=L*` 即「root 入口每帧串行接收 L* 个 flit」的最优值。

---

## 5. 与一次性贪心调度（`buildCalendarSchedule`）的关系

页面原有的一次性 makespan 贪心在 T=1 下也强制每跳 +1，但它**固定流顺序、不回溯**，
且以 makespan（注入可被 stall 推迟）为框架，因此会出现 `PE:wait` 等停顿。
这些停顿是**贪心定序 + 一次性框架的产物，并非根本不可行**：
周期相位求解器证明上述 collective 都存在 T=1 无冲突稳态日历。

---

## 6. 何时需要 T>1

- 当 `P` 被外部固定为 `< L*`（例如 color/帧预算不足）时，下界 `P ≥ L*` 被破坏，T=1 必不可行，需放宽 T 或加大 P。
- 当流量非边不相交、且相位约束在给定小 `P` 下相互矛盾（差约束不可满足）时，T>1 的每跳松弛可消解冲突。
- 对本工具内置的 5 种结构化 collective、在 `P=L*` 下，实测 **T=1 已足够**；T>1 主要用于探索 `P<L*` 的强约束或自定义流量。

---

## 7. 可复现实验

`docs/color_mesh_viz.html` 勾选 **Periodic（稳态日历）** 模式后：
指标区显示 `L*`、`Period P`、`T=1 可行?`、`minT@P=L*`、`minP@T=1` 与瓶颈 witness 边；
Router Color Calendar 矩阵与泳道按 `0..P-1` 一帧展示。

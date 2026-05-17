# Flat Butterfly vs Mesh2D 拓扑在 WSE 上的对比分析

## 1. 拓扑概述

### 1.1 Mesh2D（二维网格）

Mesh2D 是 WSE（Wafer Scale Engine）的经典拓扑选择，也是 Cerebras WSE-2/WSE-3 实际采用的互连方案。在本模拟器中，Mesh2D 将每个 Reticle 内的节点排列为 6×8 网格，每个节点与上、下、左、右最多 4 个邻居相连。

**连接规则**：节点 `(r, c)` 连接到 `(r±1, c)` 和 `(r, c±1)`（边界处截断）。

```
  0 --- 1 --- 2 --- 3 --- 4 --- 5 --- 6 --- 7
  |     |     |     |     |     |     |     |
  8 --- 9 ---10 ---11 ---12 ---13 ---14 ---15
  |     |     |     |     |     |     |     |
 16 ---17 ---18 ---19 ---20 ---21 ---22 ---23
  |     |     |     |     |     |     |     |
 24 ---25 ---26 ---27 ---28 ---29 ---30 ---31
  ...
```

**关键参数**（6×8 Reticle，含 dead/IO 节点）：


| 指标                 | 值                                  |
| ------------------ | ---------------------------------- |
| 节点度（degree）        | 2~4（边/角节点度低）                       |
| 网络直径（diameter）     | R+C-2 = 6+8-2 = **12 hops**        |
| 平均跳数               | ≈ (R+C)/3 ≈ **4.7 hops**           |
| 对分带宽（bisection BW） | min(R, C) × link_bw = **6 × 128B** |
| 路由器端口数（radix）      | **5**（4 方向 + 1 本地注入）               |
| 布线复杂度              | **最低**——仅相邻节点直连                    |


### 1.2 Flat Butterfly（扁平蝶形）

Flat Butterfly 由 Kim & Dally (ISCA 2007) 提出，核心思想是将多级 butterfly 网络"展平"到单级，利用高基数（high-radix）路由器减少跳数。在模拟器的简化实现中：

- 节点按 `group_size = √N` 分组
- **组内**：每个节点连接到同组所有其他节点（全连接）
- **组间**：每个节点连接到每个其他组中对应位置的一个代表节点

以 48 节点（6×8 Reticle）为例，`group_size = 6`（8 个组，每组 6 节点）：

```
Group 0: [0,1,2,3,4,5]  ←→ 全连接
Group 1: [6,7,8,9,10,11] ←→ 全连接
...
Group 7: [42,43,44,45,46,47] ←→ 全连接

跨组：节点 i 连接到每个其他组中的 representative = i % group_size
```

**关键参数**（48 节点）：


| 指标                 | 值                                  |
| ------------------ | ---------------------------------- |
| 节点度（degree）        | (G-1) + (N/G - 1) = 5 + 7 = **12** |
| 网络直径（diameter）     | **2 hops**（最多跨一组+目标组内）             |
| 平均跳数               | ≈ **1.9 hops**                     |
| 对分带宽（bisection BW） | 远高于 mesh——组间链路数 = N/2 × link_bw    |
| 路由器端口数（radix）      | **13**（12 网络 + 1 本地注入）             |
| 布线复杂度              | **高**——需要长距离布线                     |


### 1.3 Torus2D（二维环面，参考对比）

Torus2D 在 Mesh2D 基础上增加行列两端的环回链路，将直径减半：


| 指标    | 值                                  |
| ----- | ---------------------------------- |
| 节点度   | **4**（固定）                          |
| 网络直径  | ⌊R/2⌋ + ⌊C/2⌋ = **7 hops**         |
| 对分带宽  | 2 × min(R,C) × link_bw（mesh 的 2 倍） |
| 布线复杂度 | 中等（需环回长线）                          |


> **注意**：模拟器中 Torus2D 要求 `√N` 为整数，对 6×8=48 非完全平方数会 pad 到 49 或 64，引入虚节点。

---

## 2. 拓扑特性深度对比

### 2.1 跳数与延迟


| 拓扑             | 最大跳数 | 平均跳数 | 单跳延迟                              | 最坏情况端到端延迟 |
| -------------- | ---- | ---- | --------------------------------- | --------- |
| Mesh2D         | 12   | ~4.7 | 1 cycle (link) + 4 cycle (router) | 60 cycles |
| Flat Butterfly | 2    | ~1.9 | 1 cycle + 4 cycle                 | 10 cycles |
| Torus2D        | 7    | ~3.5 | 1 cycle + 4 cycle                 | 35 cycles |


Flat Butterfly 的延迟优势在跳数上约为 Mesh2D 的 **1/6**，但受限于：

- 高基数路由器可能增加交叉开关（crossbar）面积和仲裁延迟
- 长距离布线增加信号传播延迟（本模拟器未建模物理线延迟）

### 2.2 吞吐量与对分带宽

**对分带宽**是衡量拓扑在"最坏情况"全局通信下的吞吐量上限：


| 拓扑             | 对分带宽公式            | 48 节点值（×128B/cycle） |
| -------------- | ----------------- | ------------------- |
| Mesh2D         | min(R,C) = 6      | **768 B/cycle**     |
| Flat Butterfly | ~N/2 组间链路         | **~3072 B/cycle**   |
| Torus2D        | 2 × min(R,C) = 12 | **1536 B/cycle**    |


Flat Butterfly 的对分带宽约为 Mesh2D 的 **4 倍**，这对 allreduce 等全局通信模式非常有利。

### 2.3 VC 等待与拥塞

从 200-trial DSE 结果来看：


| 拓扑                 | 平均 vc_wait_cycles | 最小 vc_wait_cycles | 最大 vc_wait_cycles |
| ------------------ | ----------------- | ----------------- | ----------------- |
| **Flat Butterfly** | **180,579**       | **1,341**         | 1,527,936         |
| Mesh2D             | 502,849           | 13,757            | 2,545,356         |
| Torus2D            | 320,703           | 6,015             | 1,748,256         |


Flat Butterfly 的 VC 等待全面领先，最小值仅为 Mesh2D 的 **1/10**，说明其在低负载下能有效避免拥塞热点。

### 2.4 硬件成本


| 成本因素    | Mesh2D | Flat Butterfly | 影响                         |
| ------- | ------ | -------------- | -------------------------- |
| 路由器端口数  | 5      | 13             | Flat Butterfly 路由器面积 ~6.7× |
| 交叉开关复杂度 | O(5²)  | O(13²)         | 仲裁和面积急剧增长                  |
| 布线密度    | 仅相邻    | 跨组长线           | Flat Butterfly 布线 ~3×      |
| 功耗      | 低      | 高（长线驱动）        | 晶圆级长线功耗显著                  |
| 可制造性    | 极佳     | 受限             | 晶圆级长线良率下降                  |


> **WSE 的现实约束**：Cerebras WSE 实际采用 Mesh2D 的核心原因是晶圆级规模下长距离布线的物理限制。Flat Butterfly 的跨组链路在 die 间需要跨越数毫米甚至厘米，信号完整性和功耗成本极高。

---

## 3. DSE 实验结果分析

### 3.1 各拓扑最佳配置


| 拓扑                 | 最优延迟        | 最佳切分策略        | 分片数 | Batch | Tile Pipeline |
| ------------------ | ----------- | ------------- | --- | ----- | ------------- |
| **Flat Butterfly** | **105,888** | entwined_ring | 7   | 4     | False         |
| Mesh2D             | 105,934     | streaming     | 7   | 4     | False         |
| Torus2D            | 119,979     | k_split       | 7   | 4     | False         |


三者最优配置的延迟差距很小（<1%），说明本工作负载是 **memory/compute-bound** 而非 network-bound。

### 3.2 拓扑 × 切分策略交叉分析（最低延迟）


| 拓扑             | expert | row  | col  | k_split  | block | hybrid_nk | entwined_ring | streaming |
| -------------- | ------ | ---- | ---- | -------- | ----- | --------- | ------------- | --------- |
| Flat Butterfly | 707K   | 706K | 148K | **120K** | 888K  | **144K**  | **106K**      | 710K      |
| Mesh2D         | 707K   | 708K | 237K | 231K     | 354K  | **129K**  | 181K          | **106K**  |
| Torus2D        | 715K   | 707K | 877K | **120K** | 437K  | 368K      | **130K**      | 235K      |


**关键发现**：

1. **Flat Butterfly + entwined_ring** 是全局最优（105,888 cycles），因为：
  - entwined_ring 的 allreduce 折扣（0.38×）大幅降低通信尾延迟
  - Flat Butterfly 的低跳数进一步减少 VC 竞争
2. **Mesh2D + streaming** 几乎持平（105,934 cycles），因为：
  - streaming 通过 K 维流水化将内存延迟分摊到 shards 上
  - 此场景下网络通信已被流水线隐藏
3. **Mesh2D + hybrid_nk** 表现出色（129,344 cycles），因为：
  - hybrid_nk 针对 W1/W3（K>N）用 K-split，W2（N>K）用 col-split
  - 每个 GEMM 都沿更长维度切分，最小化 per-shard 内存传输
4. **Torus2D 表现不稳定**——最优点与其他拓扑一致，但平均延迟最高，因为模拟器对非完全平方节点数需要 padding。

### 3.3 Allreduce 开销分析


| 拓扑             | col 平均 allreduce | k_split 平均 allreduce | entwined_ring 平均 allreduce | hybrid_nk 平均 allreduce |
| -------------- | ---------------- | -------------------- | -------------------------- | ---------------------- |
| Flat Butterfly | 1,269            | 1,687                | **1,756**                  | 2,355                  |
| Mesh2D         | 1,402            | 515                  | **828**                    | 4,618                  |
| Torus2D        | 0                | 4,121                | **814**                    | 3,900                  |


- Flat Butterfly 在所有策略下的 allreduce 开销相对均匀，受益于高对分带宽
- entwined_ring 在三种拓扑上都大幅降低 allreduce（0.38× 折扣效果明显）
- hybrid_nk 的 allreduce 较高，因为它同时包含 K-split 和 col 两路 allreduce 通信

---

## 4. 拓扑与 FFN 切分策略的适配分析

### 4.1 DeepSeek-V4-Pro FFN 的通信模式

单专家 FFN 的 3-GEMM SwiGLU 计算图：

```
Token [B, 7168] → W1[7168, 3072] → SiLU → ⊙ → W2[3072, 7168] → Output [B, 7168]
                  W3[7168, 3072] ↗
```

不同切分策略产生的通信模式差异：


| 切分策略          | 主要通信模式            | 通信量特征               | 通信拓扑需求   |
| ------------- | ----------------- | ------------------- | -------- |
| expert        | 无 allreduce       | 仅 IO 注入/回收          | 低，任何拓扑均可 |
| row           | 无 allreduce       | 仅 IO 注入/回收          | 低，任何拓扑均可 |
| col           | N 维切分后的 allreduce | 全局归约 [B, hidden]    | 需要高对分带宽  |
| k_split       | K 维切分后的 allreduce | 归约 [B, ffn_dim] × 2 | 需要高对分带宽  |
| block         | N 维部分归约           | 部分列 allreduce       | 中等对分带宽   |
| hybrid_nk     | 混合 K + N 归约       | 最大通信量               | 需要最高对分带宽 |
| entwined_ring | 折扣 col allreduce  | 通信-计算重叠             | 受益于低延迟拓扑 |
| streaming     | K 流水 allreduce    | 归约 [B, ffn_dim] × 2 | 受益于高带宽拓扑 |


### 4.2 最佳拓扑-策略配对推荐

#### Mesh2D 最佳适配


| 推荐策略          | 原因                                 |
| ------------- | ---------------------------------- |
| **streaming** | 流水化隐藏通信，Mesh2D 的局部性足够；最优结果 105,934 |
| **hybrid_nk** | 每个 GEMM 沿最长维切分，单次 hop 内存传输最小化      |
| expert / row  | 无 allreduce 需求，Mesh2D 完全胜任         |


Mesh2D 适合通信量小或可隐藏的场景。其规则的局部互连保证了确定性延迟，且不浪费面积在长距离链路上。

#### Flat Butterfly 最佳适配


| 推荐策略              | 原因                                            |
| ----------------- | --------------------------------------------- |
| **entwined_ring** | 叠加 allreduce 折扣 + 低跳数，全局最优 105,888            |
| **col / k_split** | 需要全局 allreduce 时，高对分带宽显著降低通信延迟                |
| **hybrid_nk**     | 双路 allreduce 需要大吞吐，Flat Butterfly 的 4× 对分带宽有效 |


Flat Butterfly 适合通信密集型策略，特别是多分片 allreduce 场景。

#### 综合推荐矩阵

```
                    Memory-bound            Compute-bound           Network-bound
                    (大权重矩阵)              (小 batch)              (多分片 allreduce)
                    
Mesh2D          ✓ streaming              ✓ expert/row            ✗ 对分带宽不足
                  (流水隐藏内存延迟)        (无 allreduce 需求)       

Flat Butterfly  ✓ entwined_ring          △ expert/row            ✓✓ col/k_split/hybrid_nk
                  (折扣 allreduce)         (过度配置)               (高对分带宽匹配)

Torus2D         △ k_split                △ expert/row            △ col
                  (环回减少最坏跳数)        (无 allreduce 需求)       (中等对分带宽)
```

---

## 5. WSE 晶圆级部署的现实考量

### 5.1 物理约束


| 约束    | Mesh2D         | Flat Butterfly |
| ----- | -------------- | -------------- |
| 布线长度  | 仅相邻 die（~mm 级） | 跨组链路可达 ~cm 级   |
| 信号完整性 | 短线路优良          | 长线路需中继/增强驱动    |
| 功耗    | 低（短距离开关）       | 高（长距离驱动 ~3×）   |
| 良率影响  | 低（局部布线缺陷影响有限）  | 高（跨组链路断裂影响严重）  |
| 时钟分配  | 容易对齐           | 长线路时钟偏斜大       |


### 5.2 分层混合方案（推荐）

考虑到 WSE 的两级网络架构（NoC Reticle 内 + NoW Reticle 间），推荐的混合方案：

```
Reticle 内部 (NoC): Mesh2D
  - 44 核（6×8 - 4 dead/IO），物理尺寸 ~20mm × 25mm
  - 短距离互连，Mesh2D 完全胜任
  - 确定性 XY 路由延迟可预测

Reticle 之间 (NoW): Flat Butterfly 或 Mesh2D
  - 4 个 Reticle（2×2），物理尺寸 ~50mm × 60mm
  - 若 Reticle 间需要大量 allreduce：Flat Butterfly
  - 若通信可通过 streaming 隐藏：Mesh2D 足矣
```

### 5.3 DSE 结果验证的建议

当前 200-trial DSE 表明 Mesh2D 和 Flat Butterfly 在最优配置下几乎等效（~0.05% 差距）。这意味着：

1. **DeepSeek-V4-Pro FFN decode 工作负载是 memory/compute-bound**，拓扑选择不是主要瓶颈
2. **切分策略的影响远大于拓扑选择**——同一拓扑下最优 vs 最差策略差距可达 7×
3. **实际部署应优先选择 Mesh2D**——在几乎等效的性能下，Mesh2D 的成本和可制造性优势是决定性的

---

## 6. 结论


| 维度     | Mesh2D                | Flat Butterfly      | 推荐             |
| ------ | --------------------- | ------------------- | -------------- |
| 延迟性能   | 优（streaming 最优）       | 优（entwined_ring 最优） | 持平             |
| 吞吐量    | 中等                    | 高                   | Flat Butterfly |
| 拥塞控制   | 中等                    | 优（低 VC 等待）          | Flat Butterfly |
| 硬件成本   | **低**                 | 高                   | **Mesh2D**     |
| 可制造性   | **极佳**                | 受限                  | **Mesh2D**     |
| 功耗效率   | **优**                 | 差                   | **Mesh2D**     |
| 最佳搭配策略 | streaming / hybrid_nk | entwined_ring / col | 取决于策略          |


**总体建议**：对于 DeepSeek-V4-Pro FFN 的 WSE 部署：

- **NoC（Reticle 内）**：采用 **Mesh2D** + **streaming 或 hybrid_nk** 切分策略
- **NoW（Reticle 间）**：如果 allreduce 通信成为瓶颈，可考虑 Flat Butterfly；否则 Mesh2D 足矣
- 优先通过 **切分策略优化** 降低通信需求，而非通过增加拓扑复杂度来提升互连能力


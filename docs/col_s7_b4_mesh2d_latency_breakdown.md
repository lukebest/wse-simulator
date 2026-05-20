# Col / S7 / B4 / Mesh2D 时延分解（4-Step 模型）

本文档对配置 `col / partition_shards=7 / batch_size=4 / NoC=mesh2d` 做端到端时延分解，按用户定义的 4 个 pipeline step 重组，并与当前仿真器（`evaluate_deepseek_v4_pro_ffn`）的原始量对账，供后续分析与模型修正参考。

---

## 1. 配置与负载

| 参数 | 值 |
|------|-----|
| 分片策略 | `col`（沿 N 维列切分） |
| `partition_shards` | 7 |
| `decode_tokens` (batch B) | 4 |
| `hidden_dim` | 7168 |
| `expert_ffn_dim` | 3072 |
| 活跃 expert | 24 routed + 1 shared = **25** |
| NoC 拓扑 | `mesh2d` 6×8 |
| NoW 拓扑 | `mesh2d`（默认） |
| `tile_pipeline` | False |

单 expert SwiGLU 计算图：

```
token [B, 7168]
    ├─ W1 [7168, 3072] → SiLU ─┐
    └─ W3 [7168, 3072] ────────┼─ element-wise mul → [B, 3072]
                               │
                         W2 [3072, 7168]
                               │
                         output [B, 7168]
```

Col 切分后每 shard 有效维度（S=7）：

| 量 | 全局 | 每 shard |
|----|------|----------|
| N (ffn 中间维) | 3072 | ⌈3072/7⌉ = **439** |
| N (hidden 输出) | 7168 | ⌈7168/7⌉ = **1024** |
| K (hidden 输入) | 7168 | 7168（输入激活完整广播） |

---

## 2. 硬件与带宽假设（仿真器默认）

| 资源 | 参数 | 换算 |
|------|------|------|
| Cube | m=4, k=32, n=16 tile；startup=27；steady=5 cycles/tile | |
| 每核内存 | 256 GB/s，latency 100 ns | 256/2.0 = **128 B/cycle**；启动惩罚 **200 cycles** |
| IO 注入 | 256 GB/s @ 2.2 GHz NoC | **116.4 B/cycle** |
| NoC | mesh2d，128 B/flit，1 flit/cycle（本例默认） | |

---

## 3. 仿真器原始量（实跑 `evaluate_deepseek_v4_pro_ffn`）

复现命令（工作目录 `wse-simulator`）：

```bash
PYTHONPATH=. .venv/bin/python -c "
from copy import deepcopy
from wsesim.core.config import WSEConfig
from wsesim.dse.evaluator_deepseek import evaluate_deepseek_v4_pro_ffn
cfg = WSEConfig()
cfg.workload.decode_tokens = 4
cfg.workload.partition_strategy = 'col'
cfg.workload.partition_shards = 7
cfg.network.noc.topology = 'mesh2d'
print(evaluate_deepseek_v4_pro_ffn(cfg))
"
```

### 3.1 分阶段 compute / memory（单 expert，取 max over experts）

| 阶段 | Compute (cycles) | Memory (cycles) | max(compute, mem) |
|------|------------------|-----------------|-------------------|
| W1 | 31,382 | 49,843 | **49,843** |
| W3 | 31,382 | 49,843 | **49,843** |
| ElemMul | 1,756 | — | **1,756** |
| W2 | 30,742 | 49,608 | **49,608** |

### 3.2 通信与集合（全 workload，含 25 个活跃 expert）

| 指标 | 值 (cycles) | 说明 |
|------|-------------|------|
| `network_cycles` | 5,145（内部分析）/ 15,645（完整 evaluator 外推） | IO↔核 dispatch + combine 的 NoC 仿真外推 |
| `io_injection_cycles` | 1,355（内部分析）/ 3,942（完整 evaluator） | IO 峰值字节 / IO 带宽 |
| `allreduce_cycles` | **153,358** | SimPy ring，col 策略 payload = B×hidden×2 per expert |
| **`total_latency_cycles`** | **254,565** | max-path / stage-overlap 模型 |

### 3.3 顶层 overlap 公式（当前实现）

```
ffn_path = max(W1_stage, W3_stage) + ElemMul + W2_stage
         = max(49,843, 49,843) + 1,756 + 49,608
         = 101,207 cycles

comm_tail = max(network_cycles, io_injection_cycles, allreduce_cycles)
          = max(≈5,145, ≈1,355, 153,358)   # 内部分析量级；完整 run 见上表
          = 153,358 cycles

total = ffn_path + comm_tail = 101,207 + 153,358 = 254,565 cycles
```

W1 与 W3 在不同核上**并行**，stage 取 max 而非求和。

---

## 4. 用户 4-Step Pipeline 分解

### Step 1：输入激活 [B, 7168] 广播 + 读 W1/W3 + 计算 W1/W3

**数据流**

1. IO 注入 token `[4, 7168]`（57,344 B / expert）
2. NoC 将输入 dispatch 到各 expert 所在核
3. 每 shard 读 W1/W3 权重列块 + 输入激活，Cube 计算部分列

**访存估算（单 expert，单 shard 最慢路径）**

```
n_eff = ⌈3072/7⌉ = 439
weight_bytes = K × n_eff × 2 = 7168 × 439 × 2 = 6,293,504 B
activation_bytes = B × (K + n_eff) × 2 = 4 × (7168 + 439) × 2 = 60,856 B
mem_cycles = (weight_bytes + activation_bytes) / 128 + 200 ≈ 49,843
```

**计算估算（Cube）**

```
tiles = ⌈4/4⌉ × ⌈439/16⌉ × ⌈7168/32⌉ = 1 × 28 × 224 = 6,272
compute_cycles = 27 + (6272 - 1) × 5 = 31,382
```

**Step 1 耗时（overlap 模型）**

```
W1_stage = max(31,382, 49,843) = 49,843
W3_stage = max(31,382, 49,843) = 49,843  （与 W1 并行）
Step1 ≈ max(W1_stage, W3_stage) = 49,843 cycles   ← memory-bound
```

IO/NoC 广播（~3k cycles 量级）被 max 掩盖，未单独进入 `comm_tail`。

---

### Step 1.5：Element-wise Mul（SiLU × W3 路径）

```
ElemMul = B × n_eff = 4 × 439 = 1,756 cycles
```

串行接在 Step 1 之后（在 max(W1,W3) 之后）。

---

### Step 2：中间激活 AllGather [B, 3072]（7 shards → 完整向量）

**语义**：Col 切分后每核持有 `[B, 3072/7]` 的 partial 结果，W2 前需 all-gather 成 `[B, 3072]`。

**每 expert payload**

```
payload_mid = B × ffn_dim × fp16 = 4 × 3072 × 2 = 24,576 B
```

**按与 Step 4 相同 ring 模型缩放估算**

```
payload_final = B × hidden × 2 = 4 × 7168 × 2 = 57,344 B
Step2_estimate ≈ (24,576 / 57,344) × 153,358 ≈ 65,700 cycles
```

| 状态 | 说明 |
|------|------|
| **当前仿真器** | **未单独建模**；`_simulate_allreduce_cycles` 在 col 策略下仅使用 `B×hidden` 作为 payload |
| 若按严格 4-step 串行 | 应在 Step 1.5 与 Step 3 之间增加 ~65.7k cycles |

---

### Step 3：读 W2 + 计算 W2（输入已为 [B, 3072]）

**数据流**：AllGather 后各核有完整中间激活；每 shard 读 W2 的列块（沿 hidden N 切分）。

```
n_eff_w2 = ⌈7168/7⌉ = 1024
weight_bytes = 3072 × 1024 × 2 = 6,291,456 B
activation_bytes = 4 × (3072 + 1024) × 2 = 32,768 B
mem_cycles ≈ 49,608
compute: tiles = 1 × 64 × 224 = 6,144 → 27 + 6143×5 = 30,742

Step3 = max(30,742, 49,608) = 49,608 cycles   ← memory-bound
```

---

### Step 4：输出 AllGather [B, 7168] + NoC combine + IO eject

**数据流**

1. 各 shard 的 `[B, 7168/7]` partial 做 ring all-reduce / all-gather
2. NoC 将结果 combine 回 IO 节点
3. IO 吐出 token

**AllGather（仿真器已建模）**

```
per_expert payload = 57,344 B
25 experts × ring(7) → SimPy 仿真 → 153,358 cycles
```

**NoC combine + IO（与 Step 1 对称，量级小）**

```
≈ ½ network_cycles + ½ io_injection  （相对 allgather 可忽略）
```

```
Step4 ≈ 153,358 cycles   ← allreduce-bound
```

---

## 5. 4-Step 总览表

| Step | 操作 | Cycles | 占 total 比例 | 仿真器是否建模 |
|------|------|--------|---------------|----------------|
| 1 | 输入广播 + W1/W3 访存与计算 | 49,843 | 19.6% | ✓（IO/NoC 被 overlap 掩盖） |
| 1.5 | ElemMul | 1,756 | 0.7% | ✓ |
| 2 | 中间 AllGather `[B, 3072]` | **~65,700**（估算） | **~25.8%**（若串行叠加） | ✗ **缺失** |
| 3 | W2 访存与计算 | 49,608 | 19.5% | ✓ |
| 4 | 输出 AllGather + combine + IO | 153,358 | 60.2% | ✓（combine/IO 未单独突出） |
| **当前仿真器 total** | overlap 模型 | **254,565** | 100% | — |
| **若补 Step2 串行** | 粗算 | **~320,265** | — | 待实现 |

---

## 6. 与仿真器模型的差异（待确认 / 待改）

### 6.1 缺失 Step 2 中间 AllGather

- **现象**：col 切分在 W1/W3 后、W2 前需要一次 `[B, ffn_dim]` 的 all-gather；当前仅对最终 `[B, hidden]` 做了一次 ring 仿真。
- **影响**：低估 col 策略通信延迟约 **~65k cycles**（本配置粗算）。
- **建议**：在 `_simulate_allreduce_cycles` 中为 col 增加 `mid_allgather` payload（`B × ffn_dim × 2`），与 `final_allgather` 分别累加或取 max，按产品语义选择。

### 6.2 IO dispatch 与 combine 未拆分

- **现象**：`network_cycles` 将 IO→核（Step 1）与 核→IO（Step 4）合并仿真；`io_injection_cycles` 取 dispatch+combine 的 IO 峰值。
- **影响**：无法在本 markdown 的 Step 1/4 中分别对账独立 NoC/IO 延迟。
- **建议**：`_estimate_network_metrics` 返回 `dispatch_cycles` / `combine_cycles` 分项。

### 6.3 comm_tail 使用 `max` 而非串行求和

- **现象**：`comm_tail = max(network, io, allreduce)`，Step 4 的 allreduce 完全掩盖 Step 1 的 IO/NoC（~3k cycles）。
- **影响**：与用户 4-step **串行** 心智模型不一致；若需严格 4-step 报表，应使用 `sum` 或分 step 记账。
- **建议**：增加 `pipeline_mode: overlap | sequential` 配置项。

### 6.4 W1/W3 并行 vs 用户 step 描述

- **仿真器**：`max(W1, W3)`，符合多核并行。
- **用户 Step 1**：描述为同一 step 内完成 W1/W3，与 max 语义一致；若理解为串行 W1+W3，则会比仿真器大约 2×（~100k vs ~50k）。

---

## 7. 计算过程摘录（便于复核）

### 7.1 W1/W3 内存周期

```
mem_bw = 256 GB/s / 2.0 GHz = 128 B/cycle
weight_bytes = 7168 × 439 × 2 = 6,293,504
activation_bytes = 4 × (7168 + 439) × 2 = 60,856
transfer = (6293504 + 60856) / 128 = 49,643
total = 49,643 + 200 = 49,843 cycles
```

### 7.2 W1 Cube 计算周期

```
num_tiles = 1 × 28 × 224 = 6,272
cycles = 27 + 6271 × 5 = 31,382
```

### 7.3 AllReduce（当前实现，final only）

```
per_expert_bytes = 4 × 7168 × 2 = 57,344
active_experts = 25
mode = sequential ring on NoC, 3 experts simulated then scaled
→ 153,358 cycles (SimPy)
```

### 7.4 中间 AllGather 粗算（未实现）

```
per_expert_bytes = 4 × 3072 × 2 = 24,576
scale vs final = 24576 / 57344
estimate = 153358 × (24576/57344) ≈ 65,700 cycles
```

---

## 8. 后续分析建议

1. **实现双次 allreduce**（mid + final）后，重新跑 DSE 中 `col` 与 `entwined_ring` 排名。
2. **拆分 dispatch/combine** 网络指标，使 4-step 报表可与仿真器逐项对账。
3. **对本配置跑 200-trial**（仅 `expert/col/k_split`）时，关注 `col/s7` 是否仍为 Pareto 前沿及 allreduce 占比变化。
4. 若 Wafer 多 reticle，需在 Step 1/4 中加上 **NoW gateway** 延迟（当前已有 `gateway_noc_hops` 元数据，未并入 4-step 表）。

---

## 9. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-21 | 初版：基于 `col/s7/b4/mesh2d` 评估器实跑与 4-step 用户模型整理 |

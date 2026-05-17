# DeepSeek-V4-Pro FFN 分片策略详解

本文档描述当前仿真器支持的矩阵分片策略，以及可扩展的新策略方向。

以 DeepSeek-V4-Pro 单 expert 的 3-GEMM SwiGLU 计算图为背景：

```
token [batch, 7168]
    ├─ W1 [7168, 3072] ──→ SiLU ─┐
    │                              ├─ element-wise mul ──→ [batch, 3072]
    └─ W3 [7168, 3072] ──────────┘
                                          │
                                    W2 [3072, 7168]
                                          │
                                    output [batch, 7168]
```

GEMM 通用记号：`C[M,N] = A[M,K] × B[K,N]`

---

## 一、已实现策略

### 1. Expert Partition（专家级分片）

**核心思想**：不拆分矩阵维度，每个 expert 整体映射到一个核。

**分片方式**：
- 对 `GEMM(M, N, K)` 不做任何维度切分
- 每个 shard 拿到完整的 `(M, N, K)` 任务
- 多个 expert 之间天然并行（每个 expert 独占一个或一组核）

**实现**（`wsesim/workload/partition/expert.py`）：
```python
class ExpertPartition(PartitionStrategy):
    def partition(self, op, shards):
        return [TileTask(op_name=op.name, shard_id=s, m=op.m, n=op.n, k=op.k)
                for s in range(max(1, shards))]
```

**通信代价**：
- 无需 allreduce（每个 expert 独立计算完整结果）
- 仅需 IO dispatch/combine（token 分发与聚合）

**适用场景**：
- 核数 >= 活跃 expert 数（每个 expert 至少 1 核）
- 单 expert 工作集能放入单核 L1

**DeepSeek 具体场景**：
- 单 expert 权重 W1+W3+W2 = 3 × 7168 × 3072 × 2B ≈ **126 MB**
- 单核 L1 = 2 MB → **远超 L1 容量**，必须频繁从外部内存搬运
- 核内 memory 带宽成为瓶颈

**优缺点**：
| 优点 | 缺点 |
|------|------|
| 零通信开销（无 allreduce） | 单核承担全部权重搬运 |
| 实现最简单 | 不能利用多核分摊 memory-bound |
| expert 间完美并行 | batch 很小时 compute 利用率低 |

---

### 2. Col Partition（N 维列切分）

**核心思想**：将输出维度 N 均匀切分到多个 shard，每个 shard 计算部分列。

**分片方式**：
- `GEMM(M, N, K)` → shard_i 计算 `GEMM(M, N/S, K)`
- 每个 shard 只需要读取 B 矩阵的 `N/S` 列（权重按列分块）
- A 矩阵（输入 activation）需要完整广播到每个 shard

```
     B[K, N]
  ┌────┬────┬────┐
  │ B0 │ B1 │ B2 │  ← 按列切成 S 份
  └────┴────┴────┘
A[M,K] × B0 → C0[M, N/S]
A[M,K] × B1 → C1[M, N/S]
A[M,K] × B2 → C2[M, N/S]
拼接 → C[M, N]
```

**实现**（`wsesim/workload/partition/col.py`）：
```python
class ColPartition(PartitionStrategy):
    def partition(self, op, shards):
        cols_per_shard = ceil(op.n / max(shards, 1))
        tasks = []
        for s in range(shards):
            n_shard = max(0, min(cols_per_shard, op.n - s * cols_per_shard))
            if n_shard > 0:
                tasks.append(TileTask(..., m=op.m, n=n_shard, k=op.k))
        return tasks
```

**通信代价**：
- W1/W3 的 col-split 输出需要在 W2 之前做 allreduce（因为 W2 的输入维度要完整）
- allreduce payload = `decode_tokens × hidden_dim × 2B × active_experts`
- 使用 ring allreduce：`transfer = 2(S-1)/S × payload / noc_bytes_per_cycle`

**适用场景**：
- memory-bound 为主时，S 个 shard 分摊权重读取 → 内存墙降到 1/S
- NoC 带宽足够承受 allreduce 开销

**DeepSeek 具体场景（col/s7）**：
- 每 shard 权重：7168 × ceil(3072/7) × 2B ≈ **6.03 MB**（vs expert 模式 42 MB/矩阵）
- 内存搬运量降到 1/7，但需要一次 ring allreduce

**优缺点**：
| 优点 | 缺点 |
|------|------|
| 线性降低 per-shard 内存搬运量 | 需要 allreduce（col→W2 的拼接） |
| 每 shard 计算量均匀 | allreduce 延迟随 shard 数增加 |
| 适合 memory-bound 场景 | 输入 activation 需广播 |

---

### 3. K-split Partition（K 维切分）

**核心思想**：将规约维度 K 均匀切分，每个 shard 计算部分和，最后做 reduce-sum。

**分片方式**：
- `GEMM(M, N, K)` → shard_i 计算 `GEMM(M, N, K/S)`
- 每个 shard 产出完整形状的 `[M, N]` 部分和
- 最终需要 **allreduce（sum）** 得到完整结果

```
A[M, K] 按列切        B[K, N] 按行切
┌────┬────┬────┐      ┌────┐
│ A0 │ A1 │ A2 │      │ B0 │
└────┴────┴────┘      ├────┤
                      │ B1 │
                      ├────┤
                      │ B2 │
                      └────┘
A0 × B0 → P0[M,N]
A1 × B1 → P1[M,N]
A2 × B2 → P2[M,N]
allreduce-sum(P0, P1, P2) → C[M, N]
```

**实现**（`wsesim/workload/partition/k_split.py`）：
```python
class KPartition(PartitionStrategy):
    def partition(self, op, shards):
        k_per_shard = ceil(op.k / max(shards, 1))
        tasks = []
        for s in range(max(shards, 1)):
            k_shard = max(0, min(k_per_shard, op.k - s * k_per_shard))
            if k_shard > 0:
                tasks.append(TileTask(..., m=op.m, n=op.n, k=k_shard))
        return tasks
```

**通信代价**：
- 每个 GEMM 都需要 allreduce-sum
- allreduce payload = `2 × decode_tokens × ffn_dim × 2B × active_experts`（W1 和 W3 各一次）
- 比 col-split 多一倍 allreduce 流量（因为每个 shard 输出都是完整 [M,N]）

**适用场景**：
- K 维度远大于 N 维度时（如 W2: K=3072, N=7168）
- 需要均匀分摊权重行数

**DeepSeek 具体场景（k_split/s7）**：
- W1 每 shard：ceil(7168/7) × 3072 × 2B ≈ **6.27 MB**
- W2 每 shard：ceil(3072/7) × 7168 × 2B ≈ **6.03 MB**

**优缺点**：
| 优点 | 缺点 |
|------|------|
| 均匀切分 K 维 | allreduce 流量是 col 的 ~2 倍 |
| 适合 K >> N 的算子 | 每 shard 产出完整 [M,N]，需 sum |
| 与 col 互补 | 输出 activation 不能直接拼接 |

---

### 4. Row Partition（M 维行切分，已实现但未纳入 DSE）

**核心思想**：按 batch/token 维度切分，每个 shard 处理部分 token。

**分片方式**：
- `GEMM(M, N, K)` → shard_i 计算 `GEMM(M/S, N, K)`
- 每个 shard 需要完整的 B 矩阵（权重不分）
- 输出直接拼接，无需 allreduce

```
A[M, K] 按行切        B[K, N] 完整复制
┌────┐                 ┌────┐
│ A0 │                 │    │
├────┤  × (same B) →   │ B  │
│ A1 │                 │    │
├────┤                 └────┘
│ A2 │
└────┘
A0 × B → C0[M/S, N]
A1 × B → C1[M/S, N]
concat → C[M, N]（无需 allreduce）
```

**通信代价**：
- 无 allreduce
- 但每个 shard 都要读取完整权重 → 不降低内存搬运量

**实现**（`wsesim/workload/partition/row.py`）：
```python
class RowPartition(PartitionStrategy):
    def partition(self, op, shards):
        rows_per_shard = ceil(op.m / max(shards, 1))
        ...
        tasks.append(TileTask(..., m=m_shard, n=op.n, k=op.k))
```

**适用场景**：
- batch 很大（M >> 1）时可以有效分摊计算
- decode 场景 M 通常很小（4~16），所以该策略收益有限

**优缺点**：
| 优点 | 缺点 |
|------|------|
| 无 allreduce | 不分摊内存搬运（每 shard 读完整权重） |
| 输出直接拼接 | decode M 小时几乎无法切分 |
| 适合 prefill 大 batch | 对 memory-bound 无帮助 |

---

### 5. Block Partition（2D 块切分，已实现但未纳入 DSE）

**核心思想**：同时切 M 和 N 两个维度，形成 2D 网格分块。

**分片方式**：
- `GEMM(M, N, K)` → 按 `sqrt(S) × sqrt(S)` 网格切分
- shard(r,c) 计算 `GEMM(M/sqrt(S), N/sqrt(S), K)`
- 既分摊了计算，也分摊了 N 维度的权重读取

**实现**（`wsesim/workload/partition/block.py`）：
```python
class BlockPartition(PartitionStrategy):
    def partition(self, op, shards):
        side = max(1, isqrt(max(shards, 1)))
        block_rows = ceil(op.m / side)
        block_cols = ceil(op.n / side)
        ...
        tasks.append(TileTask(..., m=m_shard, n=n_shard, k=op.k))
```

**通信代价**：
- 需要在 M 维做 scatter（分发 activation 行）和 N 维做 reduce（列结果拼接）
- 组合了 row 和 col 的通信模式

**适用场景**：
- M 和 N 都较大时的混合切分
- decode M 很小时退化为 col 切分

**优缺点**：
| 优点 | 缺点 |
|------|------|
| 灵活利用 2D 核阵列拓扑 | 通信模式较复杂 |
| 同时分摊计算和权重 | 要求 shard 数为完全平方数效果最好 |
| 大 batch + 大模型时有优势 | decode 小 batch 下优势有限 |

---

## 二、可扩展的新策略方向

基于最新研究（MoEntwine HPCA 2026、Expert Streaming、Mozart 等），以下策略值得引入仿真：

### 6. Entwined Ring Mapping（交织环映射）

**来源**：MoEntwine (HPCA 2026)，专门针对 WSE 的 MoE 推理。

**核心思想**：将 Attention 层和 MoE 层的数据流在同一 mesh 上交织映射，使两者的通信链路互补使用。

**与当前架构的关系**：
- 当前仿真器只建模 FFN 层，Attention 层的链路在 FFN 执行时处于空闲状态
- Entwined Ring 利用这些"冷链路"来搬运 expert 权重，可隐藏通信开销
- 论文报告通信量减少 62%

**实现建议**：
- 新增 `EntwinedPartition`，在 col/k_split 基础上，引入"环状权重预取"调度
- 映射时考虑 mesh 上的物理邻居关系，让 allreduce 沿环拓扑进行
- 需要扩展仿真器以支持"跨层链路复用"建模

---

### 7. Expert Streaming / FSE-DP（全分片专家数据并行）

**来源**：Expert Streaming (2026)，针对低 batch MoE 推理。

**核心思想**：将 expert 权重按"流式"方式在多核之间传递，每个核只暂存当前正在计算的 expert 分片。

**与当前 col/k_split 的区别**：
- col/k_split 是"空间切分"——每个 shard 固定持有权重的一部分
- Expert Streaming 是"时间切分"——权重在核之间流动，每核轮流处理不同 expert 的同一分片

**优势**：
- 片上内存节省 78.8%（论文数据）
- 特别适合当前场景：单 expert 权重 126 MB >> 单核 L1 2 MB

**实现建议**：
- 新增 `StreamingPartition`，将权重沿 die-to-die 链路流水传递
- 需要扩展仿真器的时序模型，支持"计算-传输重叠"的流水线建模

---

### 8. Hybrid N-K Split（混合 NK 切分）

**核心思想**：对同一层 FFN 中不同阶段的 GEMM，根据维度比例选择不同策略。

**动机**：
- W1/W3：`[7168, 3072]`，K=7168 > N=3072 → K-split 效率更高
- W2：`[3072, 7168]`，K=3072 < N=7168 → Col-split 效率更高

**分片方式**：
- W1/W3 用 K-split
- W2 用 Col-split
- 需要在 W1/W3 → elem → W2 的衔接处做一次"通信模式切换"

**allreduce 代价**：
- W1/W3 各自 K-split 后需 reduce-sum
- W2 col-split 后需拼接
- 但每次 allreduce 的 payload 更小（因为选了各自最优的切分维度）

**实现建议**：
- 新增 `HybridNKPartition`，接受 per-op 策略映射表
- `_resolve_partitioner` 扩展为接受 `dict[op_type, strategy]`

---

### 9. Fine-Grained Pipeline Partition（细粒度流水切分）

**来源**：FinDEP (2025)、Mozart (2026)。

**核心思想**：将单个 GEMM 进一步拆成更细的 tile，以 tile 为粒度做流水调度，使计算和通信在 tile 级别重叠。

**与当前模型的关系**：
- 当前仿真器的 Cube PE 已经有 tile 概念（`4×32×16`）
- 但当前的"stage overlap"只在 W1/W3/W2 三阶段之间做重叠
- Fine-Grained Pipeline 进一步在单 GEMM 内部做 tile 级流水

**优势**：
- 能更好地隐藏 memory latency（边搬运边计算下一个 tile）
- 论文报告高达 1.61x 吞吐提升

**实现建议**：
- 不需要新的 `PartitionStrategy`，而是修改 `_estimate_compute_cycles` 中的 tile 调度模型
- 引入 tile-level overlap 系数：`effective_cycles = max(compute_tiles, memory_tiles)` 而非两者相加

---

## 三、策略对比总结

| 策略 | 切分维度 | allreduce | 权重搬运量/shard | 适用场景 | 当前状态 |
|------|---------|-----------|-----------------|---------|---------|
| **Expert** | 无 | 无 | 100% | 核数充足，compute-bound | DSE 已启用 |
| **Col** | N | 有（拼接） | 1/S | memory-bound，N 较大 | DSE 已启用 |
| **K-split** | K | 有（求和） | 1/S | memory-bound，K 较大 | DSE 已启用 |
| **Row** | M | 无 | 100% | 大 batch prefill | 已实现未启用 |
| **Block** | M+N | 有 | ~1/sqrt(S) | 大 batch + 大 N | 已实现未启用 |
| **Hybrid NK** | per-op | 有 | 1/S (最优维度) | 各 GEMM 维度不同 | 待实现 |
| **Entwined Ring** | N/K + 环映射 | 有 | 1/S + 预取隐藏 | WSE mesh 拓扑 | 待实现 |
| **Expert Streaming** | 时间维 | 有 | 流式 << 1/S | 低 batch, L1 << 权重 | 待实现 |
| **Fine-Grained Pipeline** | tile 级 | 视底层策略 | 视底层策略 | tile 级计算通信重叠 | 待实现 |

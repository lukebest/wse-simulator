# Color Mechanism Analysis (US10,515,303)

## 1. Patent summary

The Cerebras WSE fabric implements **16 logically independent networks called colors** overlaid on a single physical 2D mesh. Each color is:

1. A **virtual network** (virtual channel) with dedicated per-color buffering.
2. A **fixed static routing pattern** — no dynamic routing after compile-time configuration.
3. A **task selector** at the destination PE (`instruction_addr = base + color * 4`).

### Key hardware semantics (Router 600)

| Mechanism | Patent behavior |
|-----------|-----------------|
| `Dest[node][color]` | Static bit-vector over 7 directions (X±, Y±, skipX±, On/Off-Ramp); multiple bits = multicast replication |
| `Data Queues` | 2 entries per color; dedicated buffering, shared physical links |
| Flow control | Per-color backpressure on reverse path; queue full → stall upstream |
| Ordering | Single active input source per color; FIFO within color |
| Multicast | Router replicates wavelet to all outputs in Dest bit-vector |

### Why colors map to NN communication

Neural network connectivity is **static at compile time**. Fixed routes eliminate routing logic area and latency. Colors isolate concurrent communication patterns (e.g., partial-sum column reduce vs activation row broadcast) without dynamic arbitration.

## 2. Color scheme taxonomy

| Primitive | Topology shape | NN pattern |
|-----------|----------------|------------|
| Path/line | 1→1 fixed path (XY/YX) | Systolic streaming |
| Ring | Hamiltonian cycle | Ring all-reduce, partial-sum |
| Multicast tree | 1→N spanning tree | Activation broadcast |
| Reduction tree | N→1 converge tree | Partial-sum reduction |
| Dimension-exchange set | K colors, stride 2^k partners | RHD all-reduce, all-gather |
| All-to-all set | Time-phased permutations | MoE dispatch/combine |
| Row/column bus | Per-row X-bus, per-col Y-bus | 2D decomposed collectives |

## 3. Allocation problem

Given K colors and F concurrent flows:

- **Isolation**: overlapping flows → distinct colors
- **Reuse**: disjoint flows → same color
- **Load balance**: minimize peak link utilization → lower makespan
- **Deadlock freedom**: acyclic per-color route graph or credit-bounded ring

Strategies: greedy-by-load, graph-coloring on flow-conflict graph, ILP (pulp).

## 4. Per-pattern recommended colors

| Pattern | Primary colors | Route shape |
|---------|----------------|-------------|
| Partial-sum reduce | Column reduction-tree + ring fallback | N→1 per column |
| Activation broadcast | Row multicast-tree / row-bus | 1→N per row |
| AllReduce | Snake ring or butterfly color-set | Ring / hypercube stages |
| AllGather / ReduceScatter | Dimension-exchange + bus | Stride partners |
| All-to-all (MoE) | Time-phased permutation set | Latin-square phases |
| Systolic | Path colors along rows/cols | XY fixed paths |
| Mixed workload | Optimized K=16 allocation | Allocator output |

## 5. Simulation model mapping

```
Flit.color → per-color queue → scheduler → ColorPlan.dest[node][color] → link(s)
                                    ↑
                         per-color backpressure (reverse)
```

Ordering: `OrderTracker` records per-(src,color) sequence; violations = 0 when single-source-per-color enforced.

Fault tolerance: `color_repair.py` recomputes routes on pruned graph after `DefectMap` application.

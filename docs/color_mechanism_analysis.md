# Color Mechanism Analysis (US10,515,303)

> **Cloud SDK note:** `vendor/cerebras-cloud-sdk-python` is the Cerebras Cloud REST
> inference API and does **not** define on-wafer NoC colors. Color routing is modeled
> in `wsesim/network/` per the patent below. See `docs/color_usage_guide.md` for
> per-collective usage (broadcast, gather, reduce, allreduce, allgather).

## Core concept

A **color** is a virtual network overlaid on one physical 2D mesh. The Cerebras fabric uses ~16 colors (also 8/24/32). Each color has:

- **Dedicated per-color input buffering** at every router
- **Shared physical links** between colors
- **Fixed static routing** — no dynamic route computation at runtime
- **Independent per-color backpressure** traveling opposite to data flow

This maps NN communication staticity to hardware: since connection patterns are known at compile time, routes are fixed, eliminating dynamic routing area and latency.

## Dual role: VN ID + task selector

The wavelet **color field** selects:

1. Which virtual network (and thus which fixed route) carries the packet
2. Which compute task runs on arrival (`instruction_addr = base + color × 4`)

## Router model (Router 600)

Per node:

| Structure | Role |
|-----------|------|
| `Data Queues 650` | 2 entries × num_colors |
| `Dest 661` | Static `color → {next_hop nodes}` (multicast = multiple bits set) |
| `Sent 662` | Tracks multicast replication progress |
| `Stall Out/In` | Per-color, per-direction backpressure |

**Ordering**: one active input source per color at a time (software coordinated). Single buffer per color + fixed route ⇒ FIFO order preserved.

## Color scheme taxonomy

| Primitive | Use case | Static route shape |
|-----------|----------|-------------------|
| Path (XY/YX) | Systolic streaming | Dimension-order unicast |
| Row/column ring | Ring all-reduce, partial-sum | Fixed successor on ring |
| Snake ring | All-reduce on full mesh | Hamiltonian cycle |
| Row multicast tree | Activation broadcast | Replicate along row |
| Column reduction tree | Partial-sum reduce | Converge to column root |
| Dimension-exchange set | RHD all-reduce/gather | One color per XOR stage |
| All-to-all phases | MoE dispatch | Time-phased permutations |
| Row/column bus | 2D decomposed collectives | Shared bus per row/col |

## Allocation problem

Given K colors and a set of flows (with time windows and links used):

1. Temporally overlapping flows → **different colors**
2. Non-overlapping flows → **reuse colors**
3. Minimize **peak link load** (makespan)
4. Keep per-color route graph **acyclic** (or credit-bounded ring)

Strategies implemented: greedy-by-load, conflict-graph coloring, ILP (`pulp` with greedy fallback).

## Per-pattern color assignment (default mixed plan)

| Pattern | Color ID | Route primitive |
|---------|----------|-----------------|
| Systolic | 0 | XY path |
| Systolic alt | 1 | YX path |
| All-reduce ring | 2 | Snake ring |
| Broadcast | 3 | Row multicast |
| Partial-sum reduce | 4 | Column reduction |
| RHD stages | 5+ | Dimension exchange |
| Spare | remainder | Row/col rings or XY |

## Flow control

Per-color credits: downstream queue full ⇒ assert stall on reverse path. Queued flits at each hop form a **distributed FIFO** extending the destination queue.

## Partial-good fault tolerance

On defect:

1. Prune dead nodes/links from mesh graph
2. Recompute color routes on largest connected component
3. Preserve route *shape* where possible (tree/ring detour)
4. Metric: routable-pair coverage + makespan vs defect rate

## Simulation scope

Flit-level SimPy on **4×4** and **8×8** meshes. Baseline: single VN + XY routing. Compare makespan, link utilization, color buffer wait cycles, ordering violations.

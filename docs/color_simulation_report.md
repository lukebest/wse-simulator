# Color NoC Simulation Report

> The canonical report is now **[color_simulation_report.html](color_simulation_report.html)**
> (auto-generated from `outputs/color_vs_xy/results.csv`). This file is a text summary.

Compares the **ideal static-route color scheme** (`color_ideal`, K=24, mesh-size
independent) against a **naive single-VN dimension-order XY baseline**
(`xy_single_vn`) on 4×4 and 8×8 2D meshes. Patent background and the route
taxonomy are in [color_mechanism_analysis.md](color_mechanism_analysis.md).

## Setup

| Parameter | Value |
|-----------|-------|
| Meshes | 4×4 (16 PEs), 8×8 (64 PEs) |
| Message size | 128 bytes |
| Colors | 24 — fixed catalog of directional buses + XY/YX unicast (identical for any mesh size) |
| Baseline | Naive direct collective on one VN, XY routing |
| Workloads | `broadcast`, `gather`, `reduce`, `allreduce`, `allgather` |
| Ordering | Per `(color, src, dst)` stream lock |

Reproduce:

```bash
.venv/bin/python examples/run_color_vs_xy.py --msg-bytes 128
.venv/bin/python -m pytest tests/test_color_noc.py -q
.venv/bin/python examples/generate_color_report_html.py
```

## How colors are generated

Colors come from a **fixed, mesh-independent catalog** (`wsesim/network/color_catalog.py`,
`build_ideal_plan`). Each color is a *directional bus*: at every node its static
`dest → next_hop` points one hop in a fixed compass direction (east/west/south/north),
independent of the runtime packet destination. The same 24-color catalog is
instantiated for 4×4 and 8×8 — only the dest tables differ.

Each collective is realised (`generate_ideal_collective`) with the **minimal-link
ideal static route**:

| Workload | Ideal static route | Colors used |
|----------|-------------------|-------------|
| broadcast | spanning tree from (0,0): row-0 east ripple + per-column south ripple | `bcast_row_east`, `bcast_col_south` |
| gather | reverse tree to (0,0): columns north then row-0 west | `gather_col_north`, `gather_row_west` |
| reduce | reverse tree to (0,0), separate VN | `reduce_col_north`, `reduce_row_west` |
| allgather | 2D ring: bidirectional row pass then column pass | `allgather_row_east/west`, `allgather_col_south/north` |
| allreduce | 2D reduce-scatter + allgather along rows then columns | `allreduce_rs/ag_row_*`, `allreduce_rs/ag_col_*` |

Baseline realises each collective naively on one VN: broadcast = root→each direct,
gather/reduce = each→root direct, allgather = full all-to-all, allreduce =
reduce-to-root + broadcast.

## Results (msg = 128 bytes)

| Mesh | Pattern | XY makespan | Color makespan | Speedup |
|------|---------|-------------|----------------|---------|
| 4×4 | broadcast | 81 | 10 | **8.1×** |
| 4×4 | gather | 69 | 9 | **7.7×** |
| 4×4 | reduce | 69 | 9 | **7.7×** |
| 4×4 | allreduce | 88 | 21 | **4.2×** |
| 4×4 | allgather | 290 | 15 | **19.3×** |
| 8×8 | broadcast | 333 | 18 | **18.5×** |
| 8×8 | gather | 301 | 17 | **17.7×** |
| 8×8 | reduce | 301 | 17 | **17.7×** |
| 8×8 | allreduce | 356 | 49 | **7.3×** |
| 8×8 | allgather | 2548 | 35 | **72.8×** |

**Color ideal wins all 10 workload×mesh cases. Ordering violations = 0 everywhere.**

## Why ideal static routes win

1. **Minimal-link routing** — tree/ring uses adjacent (1-hop) edges instead of the
   baseline's many multi-hop XY unicasts converging on the root.
2. **Path diversity + VN isolation** — concurrent reduce-scatter / allgather phases
   ride dedicated row/column buses and never share a link or buffer.
3. **Pipelining** — level delays overlap tree levels / ring steps, so makespan scales
   with `(rows+cols)` rather than the participant count.
4. **Advantage grows with mesh size** — baseline root/all-to-all contention scales
   worse than tree/ring depth (largest gap: allgather 8×8, 72.8×).

## Conclusions

- Designing the ideal static routing table per collective makes the color scheme beat
  the naive single-VN XY baseline on **every** workload and both mesh sizes, with zero
  reordering.
- Color rules are **mesh-size independent** (same 24-color catalog for 4×4 and 8×8).
- Fault-tolerance code exists in `wsesim/network/color_repair.py` (not yet wired into
  this benchmark harness).

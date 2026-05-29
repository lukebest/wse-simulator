# Color NoC Simulation Report

Simulation comparing **mixed color scheme** (`color_mixed`, K=16) against **single-VN dimension-order XY** (`xy_single_vn`) on 4×4 and 8×8 2D meshes. See [color_mechanism_analysis.md](color_mechanism_analysis.md) for patent background and route taxonomy.

## Setup

| Parameter | Value |
|-----------|-------|
| Meshes | 4×4 (16 PEs), 8×8 (64 PEs) |
| Message size | 128 bytes |
| Colors | 16 (mixed plan: unicast XY/YX, row multicast, col reduction, dim-exchange, all-to-all phases) |
| Baseline | One virtual network, XY routing, shared link FC |
| Workloads | `ring`, `direct_allgather`, `broadcast_tree`, `reduction_tree`, `all_to_all`, `systolic`, `mixed` |
| Ordering model | Per `(color, src, dst)` stream lock; zero reordering violations in all runs |

Reproduce:

```bash
.venv/bin/python examples/run_color_vs_xy.py --msg-bytes 128
.venv/bin/python -m pytest tests/test_color_noc.py -q
```

Results: `outputs/color_vs_xy/results.csv`

## Summary

| Mesh | Pattern | XY makespan | Color makespan | Winner | Speedup |
|------|---------|-------------|----------------|--------|---------|
| 4×4 | ring | 469 | 720 | XY | 0.65× |
| 4×4 | direct_allgather | 290 | 46 | **Color** | **6.3×** |
| 4×4 | broadcast_tree | 19 | 17 | Color | 1.1× |
| 4×4 | reduction_tree | 15 | 15 | tie | 1.0× |
| 4×4 | all_to_all | 310 | 252 | Color | 1.2× |
| 4×4 | systolic | 21 | 9 | **Color** | **2.3×** |
| 4×4 | mixed | 1253 | 1164 | Color | 1.1× |
| 8×8 | ring | 1941 | 7056 | XY | 0.28× |
| 8×8 | direct_allgather | 2548 | 272 | **Color** | **9.4×** |
| 8×8 | broadcast_tree | 39 | 41 | XY | 0.95× |
| 8×8 | reduction_tree | 35 | 35 | tie | 1.0× |
| 8×8 | all_to_all | 4056 | 4056 | tie | 1.0× |
| 8×8 | systolic | 21 | 17 | Color | 1.2× |
| 8×8 | mixed | 16344 | 16344 | tie | 1.0× |

**Ordering violations: 0** across all 28 runs.

## Where color wins

### Direct all-gather (largest win)

Color assigns dedicated unicast paths (XY + YX colors) so many pairwise exchanges proceed in parallel without single-VN contention. On 8×8, makespan drops from **2548 → 272 cycles** (~9.4×). Average latency also improves dramatically (1254 → 144 cycles).

Trade-off: high `link_wait_cycles` on color runs (e.g. 9.3M on 8×8) reflects per-color credit backpressure under heavy parallel load — the fabric is utilized (~71% avg link util vs ~8% for XY) rather than idle.

### Systolic streaming

Dedicated path colors let each PE stream to its neighbor without competing with unrelated traffic. **4×4: 21 → 9 cycles**; color sends fewer total flits (12 vs 48) because routes are direct rather than multi-hop XY fan-out.

### All-to-all (4×4 only)

Time-phased color assignment reduces head-of-line blocking. **310 → 252 cycles** on 4×4. At 8×8 the benefit is absorbed by link saturation — both schemes finish in 4056 cycles.

### Mixed taxonomy (4×4)

Composite workload combining multiple primitives: **1253 → 1164 cycles** (~7% faster). Average latency drops from 120 → 12 cycles because concurrent color-isolated flows avoid XY serialization.

## Where color loses or ties

### Ring all-reduce

Color **loses badly** on ring: 4×4 (720 vs 469), 8×8 (7056 vs 1941). Root cause: ring traffic is currently mapped to generic unicast XY/YX colors rather than a physical ring color (mesh-adjacent ring successors are not always valid on a 2D torus without snake routing). XY baseline naturally follows short Manhattan paths for reduce-scatter / allgather phases, while our color assignment serializes or detours ring pairs.

**Note:** Average per-packet latency is much lower for color on ring (7.5 vs 148 on 4×4) because individual packets traverse fewer hops once injected; makespan is dominated by injection scheduling and lack of ring-native routing.

### Tree collectives

Broadcast and reduction trees **tie or differ by ≤2 cycles** — both schemes use similar hop counts on small meshes; multicast replication via Dest bit-vectors does not yet beat simple unicast XY for shallow trees.

### 8×8 mixed / all-to-all

Makespan **ties at 16344 / 4056** cycles. Color still cuts average latency (~542 → 45 on mixed) but global completion is link-bandwidth bound. Further wins would require more aggressive color allocation (ILP optimizer) or wider effective bandwidth via path diversity.

## Metrics interpretation

| Metric | Color typical | XY typical | Notes |
|--------|---------------|------------|-------|
| `makespan_cycles` | Lower for gather/systolic | Lower for ring | Primary DSE objective |
| `avg_latency` | Much lower when parallel | Higher under contention | Per-flit delivery time |
| `avg_link_util` | Higher under load | Lower (serialized) | Color fills links |
| `link_wait_cycles` | High on heavy parallel | Often 0 | Per-color FC backpressure |
| `ordering_violations` | 0 | 0 | `(color, src, dst)` stream lock |

## Fault tolerance (implemented, not benchmarked here)

`wsesim/network/color_repair.py` recomputes `ColorPlan` on a pruned graph from `DefectMap` and reports routable coverage. A defect-rate sweep is a natural follow-on experiment:

```python
from wsesim.network.color_repair import repair_plan, coverage_metric
# ... supply defect map, re-run compare_pattern()
```

## Conclusions

1. **Static color routing proves substantial makespan advantage** for direct all-gather and systolic patterns (up to **9.4×** on 8×8 gather) while preserving **zero packet reordering**.
2. **Ring all-reduce remains a regression** until ring-native or snake-ring colors are wired to mesh-valid successor sets and paired with correct traffic-to-color mapping.
3. **At scale, link saturation equalizes makespan** for all-to-all and mixed workloads even though color reduces average latency; the benefit shifts from completion time to responsiveness.
4. **Per-color flow control works** — backpressure appears as `link_wait_cycles` rather than buffer overflow or ordering violations.

## Recommended next steps

- Fix ring color mapping: use `snake_ring` or phase-split reduce-scatter colors with acyclic check
- Run defect-rate sweep with `color_repair` in `color_sim.py`
- Wire `color_scheme` / `num_colors` into DSE evaluator for makespan-aware search
- Compare greedy vs ILP color allocation on 8×8 mixed workload

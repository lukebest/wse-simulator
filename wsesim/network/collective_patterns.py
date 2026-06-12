"""Collective traffic patterns for 2D mesh time-expanded (bufferless) scheduling.

Mirrors flow generators in docs/color_mesh_viz.html and adds a Hamiltonian-ring
allgather pattern that achieves makespan ceil((N-1)/2) with zero stall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

PE_INOUT_LATENCY = 10


@dataclass(frozen=True, slots=True)
class MeshNode:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class MeshEdge:
    id: str
    from_node: MeshNode
    to_node: MeshNode


@dataclass(slots=True)
class CollectiveFlow:
    id: str
    from_node: MeshNode
    to_node: MeshNode
    path: list[MeshEdge]
    flits: int = 1
    release: int = 0
    color_id: int = 0
    payload: str = "allgather"
    label: str = ""
    slots: list[int] | None = None


@dataclass(slots=True)
class EdgeLoad:
    l_star: int
    witness: str | None
    edge_count: int


@dataclass(slots=True)
class ScheduleResult:
    ok: bool
    makespan: int
    total_stall: int
    peak: int
    collision: str | None = None


def node(x: int, y: int) -> MeshNode:
    return MeshNode(x, y)


def idx_to_node(idx: int, cols: int) -> MeshNode:
    return MeshNode(idx % cols, idx // cols)


def node_idx(n: MeshNode, cols: int) -> int:
    return n.y * cols + n.x


def eid(a: MeshNode, b: MeshNode) -> str:
    return f"{a.x},{a.y}->{b.x},{b.y}"


def make_edge(a: MeshNode, b: MeshNode) -> MeshEdge:
    return MeshEdge(eid(a, b), a, b)


def xy_path(a: MeshNode, b: MeshNode) -> list[MeshEdge]:
    path: list[MeshEdge] = []
    c = MeshNode(a.x, a.y)
    while c.x != b.x:
        nxt = MeshNode(c.x + (1 if b.x > c.x else -1), c.y)
        path.append(make_edge(c, nxt))
        c = nxt
    while c.y != b.y:
        nxt = MeshNode(c.x, c.y + (1 if b.y > c.y else -1))
        path.append(make_edge(c, nxt))
        c = nxt
    return path


def pow2_strides(n: int) -> list[int]:
    out: list[int] = []
    s = 1
    while s < n:
        out.append(s)
        s <<= 1
    return out


def dim_exchange_stage_cycles(stride: int, flits: int) -> int:
    return stride + max(0, flits - 1)


def compute_edge_load(flows: Iterable[CollectiveFlow]) -> EdgeLoad:
    counts: dict[str, int] = {}
    for flow in flows:
        for edge in flow.path:
            counts[edge.id] = counts.get(edge.id, 0) + 1
    if not counts:
        return EdgeLoad(l_star=1, witness=None, edge_count=0)
    witness, l_star = max(counts.items(), key=lambda kv: kv[1])
    return EdgeLoad(l_star=max(1, l_star), witness=witness, edge_count=len(counts))


def optimal_hamilton_makespan(node_count: int) -> int:
    return math.ceil((node_count - 1) / 2)


def build_hamilton_cycle(rows: int, cols: int) -> list[MeshNode]:
    """Snake Hamiltonian cycle on an open mesh; requires even X (cols)."""
    if cols % 2 != 0:
        raise ValueError(f"Hamiltonian snake requires even cols, got {cols}")
    cycle: list[MeshNode] = []
    visited: set[tuple[int, int]] = set()
    for x in range(cols):
        if x % 2 == 0:
            # Even columns: y=0..Y-1 on col 0; later even cols start at y=1 so
            # the link from the prior odd col ending at (x-1,1) is (x-1,1)->(x,1).
            y_start = 0 if x == 0 else 1
            for y in range(y_start, rows):
                cycle.append(node(x, y))
                visited.add((x, y))
        else:
            for y in range(rows - 1, 0, -1):
                cycle.append(node(x, y))
                visited.add((x, y))
    # Close along row y=0, skipping nodes already visited on even columns.
    if (cols - 1, 0) not in visited:
        cycle.append(node(cols - 1, 0))
        visited.add((cols - 1, 0))
    for x in range(cols - 2, -1, -1):
        if (x, 0) not in visited:
            cycle.append(node(x, 0))
            visited.add((x, 0))
    if len(cycle) != rows * cols:
        raise ValueError(
            f"Hamiltonian cycle length {len(cycle)} != {rows * cols} for {rows}x{cols}"
        )
    return cycle


def _ring_edges(cycle: list[MeshNode]) -> list[MeshEdge]:
    n = len(cycle)
    return [make_edge(cycle[i], cycle[(i + 1) % n]) for i in range(n)]


def build_hamilton_ring_allgather_flows(
    rows: int,
    cols: int,
    base_flits: int = 1,
) -> tuple[list[CollectiveFlow], int, dict[str, int]]:
    """Bidirectional Hamiltonian-ring pipeline allgather.

    All sources inject at slot 0. Each directed ring edge carries one flit per
    slot; makespan is ceil((N-1)/2), which is optimal on degree-2 corners.
    """
    cycle = build_hamilton_cycle(rows, cols)
    n = len(cycle)
    t_star = optimal_hamilton_makespan(n)
    pos = {cycle[i]: i for i in range(n)}
    ring_edges = _ring_edges(cycle)
    flows: list[CollectiveFlow] = []
    cid = 0

    for src_idx in range(n):
        src_node = cycle[src_idx]
        pos_src = pos[src_node]

        cw_path: list[MeshEdge] = []
        cw_slots: list[int] = []
        for hop in range(t_star):
            ring_pos = (pos_src + hop) % n
            cw_path.append(ring_edges[ring_pos])
            cw_slots.append(hop)
        flows.append(
            CollectiveFlow(
                id=f"hr_cw_{src_idx}",
                label=f"hr cw src={src_idx}",
                from_node=src_node,
                to_node=cw_path[-1].to_node if cw_path else src_node,
                path=cw_path,
                flits=base_flits,
                release=0,
                color_id=cid,
                payload="allgather_hamilton",
                slots=cw_slots,
            )
        )
        cid += 1

        ccw_path: list[MeshEdge] = []
        ccw_slots: list[int] = []
        for hop in range(t_star):
            ring_pos = (pos_src - hop) % n
            ccw_path.append(
                make_edge(cycle[ring_pos], cycle[(ring_pos - 1) % n])
            )
            ccw_slots.append(hop)
        flows.append(
            CollectiveFlow(
                id=f"hr_ccw_{src_idx}",
                label=f"hr ccw src={src_idx}",
                from_node=src_node,
                to_node=ccw_path[-1].to_node if ccw_path else src_node,
                path=ccw_path,
                flits=base_flits,
                release=0,
                color_id=cid,
                payload="allgather_hamilton",
                slots=ccw_slots,
            )
        )
        cid += 1

    meta = {
        "node_count": n,
        "t_star": t_star,
        "flow_count": len(flows),
        "cw_edges_per_slot": n,
        "ccw_edges_per_slot": n,
    }
    return flows, t_star, meta


def _transposed_hamilton_cycle(rows: int, cols: int) -> list[MeshNode]:
    """Row-major snake: Hamilton cycle of the transposed grid mapped back."""
    return [node(nd.y, nd.x) for nd in build_hamilton_cycle(cols, rows)]


def build_hamilton_ring_allgather_flows_aniso(
    rows: int,
    cols: int,
    hx: int,
    hy: int,
    base_flits: int = 1,
    orient: str = "auto",
) -> tuple[list[CollectiveFlow], int, dict[str, int]]:
    """Hamilton-ring allgather under per-axis link latency (hx, hy).

    Timing is cumulative-weighted: a flit forwards exactly when it arrives, so
    its launch on the k-th ring edge is the sum of latencies of the previous k
    edges. Conflict-freedom: on a directed ring edge, the flits crossing it
    traveled k = 1, 2, ... consecutive edges ending there; their launch times
    are suffix sums of >= 1 latencies, strictly increasing in k, hence distinct.

    orient: "col" = column-major snake (vertical-heavy), "row" = row-major
    snake (horizontal-heavy), "auto" = pick the smaller weighted makespan.
    Makespan = worst window of ceil((N-1)/2) consecutive edge latencies.
    """

    def cycle_for(o: str) -> list[MeshNode]:
        return (
            build_hamilton_cycle(rows, cols)
            if o == "col"
            else _transposed_hamilton_cycle(rows, cols)
        )

    def ring_makespan(cycle: list[MeshNode]) -> int:
        n = len(cycle)
        t_star = optimal_hamilton_makespan(n)
        lats = [
            edge_hop_latency(e, hx, hy) for e in _ring_edges(cycle)
        ]
        worst = 0
        for start in range(n):
            worst = max(
                worst, sum(lats[(start + i) % n] for i in range(t_star))
            )
        return worst

    if orient == "auto":
        candidates = {}
        for o in ("col", "row"):
            try:
                candidates[o] = ring_makespan(cycle_for(o))
            except ValueError:
                pass  # snake needs the swept dimension even
        if not candidates:
            raise ValueError(f"no Hamilton snake for {rows}x{cols}")
        orient = min(candidates, key=candidates.get)

    cycle = cycle_for(orient)
    n = len(cycle)
    t_star = optimal_hamilton_makespan(n)
    pos = {cycle[i]: i for i in range(n)}
    ring_edges = _ring_edges(cycle)
    flows: list[CollectiveFlow] = []
    cid = 0
    arrival_makespan = 0

    for src_idx in range(n):
        src_node = cycle[src_idx]
        pos_src = pos[src_node]

        for direction in ("cw", "ccw"):
            path: list[MeshEdge] = []
            slots: list[int] = []
            t = 0
            for hop in range(t_star):
                if direction == "cw":
                    edge = ring_edges[(pos_src + hop) % n]
                else:
                    ring_pos = (pos_src - hop) % n
                    edge = make_edge(cycle[ring_pos], cycle[(ring_pos - 1) % n])
                path.append(edge)
                slots.append(t)
                t += edge_hop_latency(edge, hx, hy)
            arrival_makespan = max(arrival_makespan, t)
            flows.append(
                CollectiveFlow(
                    id=f"hra_{direction}_{src_idx}",
                    label=f"hr-aniso {direction} src={src_idx}",
                    from_node=src_node,
                    to_node=path[-1].to_node if path else src_node,
                    path=path,
                    flits=base_flits,
                    release=0,
                    color_id=cid,
                    payload="allgather_hamilton_aniso",
                    slots=slots,
                )
            )
            cid += 1

    meta = {
        "node_count": n,
        "t_star": t_star,
        "flow_count": len(flows),
        "orient": orient,
        "arrival_makespan": arrival_makespan,
    }
    return flows, arrival_makespan, meta


def _add_dim_exchange_pair(
    flows: list[CollectiveFlow],
    cid_ref: list[int],
    axis: str,
    x: int,
    y: int,
    stride: int,
    flits: int,
    release: int,
    payload: str,
    id_prefix: str,
    tag: str,
    cols: int,
) -> None:
    frm = node(x, y)
    to = node(x + stride, y) if axis == "x" else node(x, y + stride)
    p_ab = xy_path(frm, to)
    p_ba = xy_path(to, frm)
    cid = cid_ref[0]
    flows.append(
        CollectiveFlow(
            id=f"{id_prefix}_{axis}{stride}_ab_{x}_{y}",
            label=f"c{cid} {tag} ({frm.x},{frm.y})->({to.x},{to.y})",
            from_node=frm,
            to_node=to,
            path=p_ab,
            flits=flits,
            release=release,
            color_id=cid,
            payload=payload,
        )
    )
    flows.append(
        CollectiveFlow(
            id=f"{id_prefix}_{axis}{stride}_ba_{x}_{y}",
            label=f"c{cid + 1} {tag} ({to.x},{to.y})->({frm.x},{frm.y})",
            from_node=to,
            to_node=frm,
            path=p_ba,
            flits=flits,
            release=release,
            color_id=cid + 1,
            payload=payload,
        )
    )
    cid_ref[0] += 2


def build_dimwise_allgather_flows(
    rows: int,
    cols: int,
    base_flits: int = 1,
    *,
    pe_switch: bool = True,
) -> list[CollectiveFlow]:
    """Dimension-wise RHD allgather (reverse strides), ported from color_mesh_viz.html."""
    flows: list[CollectiveFlow] = []
    cid_ref = [0]
    payload = "allgather"
    release = 0
    blocks = 1

    for stride in reversed(pow2_strides(cols)):
        flits = base_flits * blocks
        tag = f"X AG s={stride} x{blocks}blk"
        for y in range(rows):
            for x in range(0, cols, 2 * stride):
                for d in range(stride):
                    if x + d + stride < cols:
                        _add_dim_exchange_pair(
                            flows,
                            cid_ref,
                            "x",
                            x + d,
                            y,
                            stride,
                            flits,
                            release,
                            payload,
                            "ag",
                            tag,
                            cols,
                        )
        release += dim_exchange_stage_cycles(stride, flits)
        blocks *= 2

    if cols > 1 and rows > 1 and pe_switch:
        release += PE_INOUT_LATENCY

    blocks = cols
    for stride in reversed(pow2_strides(rows)):
        flits = base_flits * blocks
        tag = f"Y AG s={stride} x{blocks}blk"
        for x in range(cols):
            for y in range(0, rows, 2 * stride):
                for d in range(stride):
                    if y + d + stride < rows:
                        _add_dim_exchange_pair(
                            flows,
                            cid_ref,
                            "y",
                            x,
                            y + d,
                            stride,
                            flits,
                            release,
                            payload,
                            "ag",
                            tag,
                            cols,
                        )
        release += dim_exchange_stage_cycles(stride, flits)
        blocks *= 2

    return flows


def verify_preassigned_slots(flows: Iterable[CollectiveFlow]) -> ScheduleResult:
    seen: set[tuple[str, int]] = set()
    makespan = 0
    for flow in flows:
        if not flow.path:
            continue
        if flow.slots is None or len(flow.slots) != len(flow.path):
            return ScheduleResult(
                ok=False,
                makespan=0,
                total_stall=0,
                peak=0,
                collision=f"missing slots for {flow.id}",
            )
        for hop, edge in enumerate(flow.path):
            slot = flow.slots[hop]
            key = (edge.id, slot)
            if key in seen:
                return ScheduleResult(
                    ok=False,
                    makespan=0,
                    total_stall=0,
                    peak=0,
                    collision=f"{edge.id}@{slot}",
                )
            seen.add(key)
            makespan = max(makespan, slot + 1)
    peak = _peak_from_assignments(flows)
    return ScheduleResult(ok=True, makespan=makespan, total_stall=0, peak=peak)


def _peak_from_assignments(flows: Iterable[CollectiveFlow]) -> int:
    per_slot: dict[int, dict[str, int]] = {}
    for flow in flows:
        if flow.slots is None:
            continue
        for hop, edge in enumerate(flow.path):
            slot = flow.slots[hop]
            bucket = per_slot.setdefault(slot, {})
            bucket[edge.id] = bucket.get(edge.id, 0) + 1
    if not per_slot:
        return 0
    return max(max(bucket.values()) for bucket in per_slot.values())


def schedule_bufferless_noc(flows: list[CollectiveFlow]) -> ScheduleResult:
    """Rigid +1/hop bufferless placement (planNoc in color_mesh_viz.html)."""
    reserved: set[tuple[str, int]] = set()
    total_stall = 0
    peak = 0
    makespan = 0

    ordered = sorted(
        flows,
        key=lambda f: (f.release, len(f.path), f.color_id),
    )

    for flow in ordered:
        prev_first_edge = -1
        for flit in range(flow.flits):
            release_at = flow.release + flit
            if not flow.path:
                continue
            plan = _plan_noc(flow.path, release_at, prev_first_edge, reserved)
            if plan is None:
                return ScheduleResult(
                    ok=False,
                    makespan=0,
                    total_stall=total_stall,
                    peak=peak,
                    collision=f"failed {flow.id} flit {flit}",
                )
            times, edges = plan
            prev_first_edge = times[0]
            total_stall += times[0] - release_at
            flow_slots = flow.slots
            if flow_slots is None:
                flow.slots = [-1] * len(flow.path)
                flow_slots = flow.slots
            for hop, t in enumerate(times):
                reserved.add((edges[hop].id, t))
                if hop < len(flow_slots):
                    flow_slots[hop] = t
                slot_counts: dict[str, int] = {}
                for e, tt in zip(edges, times, strict=False):
                    if tt == t:
                        slot_counts[e.id] = slot_counts.get(e.id, 0) + 1
                if slot_counts:
                    peak = max(peak, max(slot_counts.values()))
                makespan = max(makespan, t + 1)

    return ScheduleResult(
        ok=True,
        makespan=makespan,
        total_stall=total_stall,
        peak=max(peak, 1),
    )


def _earliest_edge(
    start: int,
    edge_id: str,
    reserved: set[tuple[str, int]],
    limit: int = 200_000,
) -> int:
    for t in range(start, limit):
        if (edge_id, t) not in reserved:
            return t
    return limit


def _plan_noc(
    path: list[MeshEdge],
    release_at: int,
    prev_first_edge: int,
    reserved: set[tuple[str, int]],
) -> tuple[list[int], list[MeshEdge]] | None:
    max_attempts = max(8192, len(path) * 128)
    for stall in range(max_attempts):
        times: list[int] = []
        t0 = max(release_at + stall, prev_first_edge + 1 if prev_first_edge >= 0 else release_at + stall)
        t0 = _earliest_edge(t0, path[0].id, reserved)
        times.append(t0)
        ok = True
        for hop in range(1, len(path)):
            t = times[hop - 1] + 1
            if (path[hop].id, t) in reserved:
                ok = False
                break
            times.append(t)
        if ok:
            return times, path
    return None


def compare_allgather_patterns(rows: int, cols: int, base_flits: int = 1) -> dict:
    n = rows * cols
    t_star = optimal_hamilton_makespan(n)

    ham_flows, _, ham_meta = build_hamilton_ring_allgather_flows(rows, cols, base_flits)
    ham_verify = verify_preassigned_slots(ham_flows)

    dim_flows = build_dimwise_allgather_flows(rows, cols, base_flits)
    dim_load = compute_edge_load(dim_flows)
    dim_schedule = schedule_bufferless_noc(dim_flows)

    return {
        "rows": rows,
        "cols": cols,
        "node_count": n,
        "t_star": t_star,
        "hamilton": {
            "flow_count": len(ham_flows),
            "makespan": ham_verify.makespan,
            "stall": ham_verify.total_stall,
            "ok": ham_verify.ok,
            "peak": ham_verify.peak,
            "meta": ham_meta,
        },
        "dimwise": {
            "flow_count": len(dim_flows),
            "l_star": dim_load.l_star,
            "makespan": dim_schedule.makespan,
            "stall": dim_schedule.total_stall,
            "ok": dim_schedule.ok,
            "peak": dim_schedule.peak,
        },
        "speedup_vs_dimwise": (
            dim_schedule.makespan / ham_verify.makespan
            if ham_verify.makespan and dim_schedule.makespan
            else None
        ),
    }


def generate_collective_flows(
    pattern: str,
    rows: int,
    cols: int,
    flits: int = 1,
    *,
    variant: str = "dimwise",
    root: int = 0,
) -> list[CollectiveFlow]:
    pattern = pattern.lower()
    builders = {
        "broadcast": lambda: build_multicast_broadcast_flows(rows, cols, root, flits),
        "reduce": lambda: build_tree_reduce_flows(rows, cols, root, flits),
        "gather": lambda: build_gather_to_root_flows(rows, cols, root, flits),
        "allreduce": lambda: build_dimwise_allreduce_flows(rows, cols, flits, pe_switch=False),
        "alltoall": lambda: build_allto_all_flows(rows, cols, flits),
        "allgather": lambda: (
            build_hamilton_ring_allgather_flows(rows, cols, flits)[0]
            if variant == "hamilton"
            else build_dimwise_allgather_flows(rows, cols, flits, pe_switch=False)
        ),
    }
    if pattern not in builders:
        raise ValueError(f"Unsupported pattern {pattern}")
    return builders[pattern]()


# ---------------------------------------------------------------------------
# Mesh metrics & lower bounds (inline-reduce / fan-out model)
# ---------------------------------------------------------------------------


def mesh_diameter(rows: int, cols: int) -> int:
    return rows + cols - 2


def root_coords(root: int, cols: int) -> tuple[int, int]:
    return root % cols, root // cols


def node_degree(rows: int, cols: int, root: int) -> int:
    x, y = root_coords(root, cols)
    deg = 0
    if x > 0:
        deg += 1
    if x < cols - 1:
        deg += 1
    if y > 0:
        deg += 1
    if y < rows - 1:
        deg += 1
    return deg


def eccentricity(rows: int, cols: int, root: int) -> int:
    rx, ry = root_coords(root, cols)
    farthest = 0
    for y in range(rows):
        for x in range(cols):
            farthest = max(farthest, abs(x - rx) + abs(y - ry))
    return farthest


def optimal_z_lower_bound(
    pattern: str,
    rows: int,
    cols: int,
    *,
    root: int = 0,
) -> int:
    """Minimum makespan lower bound z* under inline-reduce / fan-out model."""
    n = rows * cols
    pattern = pattern.lower()
    ecc = eccentricity(rows, cols, root)
    deg = node_degree(rows, cols, root)
    d = mesh_diameter(rows, cols)

    if pattern == "broadcast":
        return ecc
    if pattern == "reduce":
        return ecc
    if pattern == "allreduce":
        return d
    if pattern in ("allgather", "gather"):
        return math.ceil((n - 1) / deg)
    if pattern == "alltoall":
        return max(
            math.ceil(n * cols / 4),
            math.ceil(n * rows / 4),
        )
    raise ValueError(f"Unknown pattern {pattern}")


def optimal_z_lower_bound_h(
    pattern: str,
    rows: int,
    cols: int,
    *,
    root: int = 0,
    hop_latency: int = 1,
) -> int:
    """Makespan lower bound with per-hop link latency h (pipelined, II=1).

    Distance terms scale by h (a flit crossing k links needs k*h cycles);
    bandwidth/serialization terms do NOT scale (each link still launches one
    flit per cycle). z*_h = max(h * distance_term, bandwidth_term).
    """
    n = rows * cols
    h = max(1, hop_latency)
    pattern = pattern.lower()
    ecc = eccentricity(rows, cols, root)
    deg = node_degree(rows, cols, root)
    d = mesh_diameter(rows, cols)

    if pattern in ("broadcast", "reduce"):
        return h * ecc
    if pattern == "allreduce":
        return h * d
    if pattern == "allgather":
        # every node must receive the farthest block (h*D) and the worst node
        # ingests N-1 blocks over deg_min=2 ports (corner) at 1 flit/cycle.
        return max(h * d, math.ceil((n - 1) / 2))
    if pattern == "gather":
        return max(h * ecc, math.ceil((n - 1) / deg))
    if pattern == "alltoall":
        return max(
            math.ceil(n * cols / 4),
            math.ceil(n * rows / 4),
            h * d,
        )
    raise ValueError(f"Unknown pattern {pattern}")


def optimal_z_lower_bound_aniso(
    pattern: str,
    rows: int,
    cols: int,
    *,
    root: int = 0,
    hx: int = 1,
    hy: int = 1,
) -> int:
    """Makespan lower bound with per-axis link latency (hx horizontal, hy vertical).

    Distance terms use the weighted Manhattan metric hx*|dx| + hy*|dy|;
    bandwidth/serialization terms are latency-free (links pipelined at II=1).
    """
    n = rows * cols
    pattern = pattern.lower()
    ecc_w = weighted_eccentricity(rows, cols, root, hx, hy)
    deg = node_degree(rows, cols, root)
    d_w = weighted_diameter(rows, cols, hx, hy)

    if pattern in ("broadcast", "reduce"):
        return ecc_w
    if pattern == "allreduce":
        return d_w
    if pattern == "allgather":
        return max(d_w, math.ceil((n - 1) / 2))
    if pattern == "gather":
        return max(ecc_w, math.ceil((n - 1) / deg))
    if pattern == "alltoall":
        return max(
            math.ceil(n * cols / 4),
            math.ceil(n * rows / 4),
            d_w,
        )
    raise ValueError(f"Unknown pattern {pattern}")


# ---------------------------------------------------------------------------
# Flow builders (ported from color_mesh_viz.html)
# ---------------------------------------------------------------------------


def _make_hop_flow(
    flows: list[CollectiveFlow],
    cid: int,
    fid: str,
    label: str,
    frm: MeshNode,
    to: MeshNode,
    flits: int,
    release: int,
    payload: str,
) -> int:
    flows.append(
        CollectiveFlow(
            id=fid,
            label=label,
            from_node=frm,
            to_node=to,
            path=[make_edge(frm, to)],
            flits=flits,
            release=release,
            color_id=cid,
            payload=payload,
        )
    )
    return cid + 1


def build_multicast_broadcast_flows(
    rows: int,
    cols: int,
    root: int,
    flits: int = 1,
) -> list[CollectiveFlow]:
    rx, ry = root_coords(root, cols)
    flows: list[CollectiveFlow] = []
    row_path: list[MeshEdge] = []
    if rx < cols - 1:
        for c in range(rx, cols - 1):
            row_path.append(make_edge(node(c, ry), node(c + 1, ry)))
    elif rx > 0:
        for c in range(rx, 0, -1):
            row_path.append(make_edge(node(c, ry), node(c - 1, ry)))
    cid = 0
    if row_path:
        flows.append(
            CollectiveFlow(
                id="row",
                label="row multicast bus",
                from_node=node(rx, ry),
                to_node=row_path[-1].to_node,
                path=row_path,
                flits=flits,
                release=0,
                color_id=cid,
                payload="broadcast",
            )
        )
        cid += 1
    base = max(1, cols - 1)
    south = ry < rows - 1
    for c in range(cols):
        col_path: list[MeshEdge] = []
        if south:
            for r in range(ry, rows - 1):
                col_path.append(make_edge(node(c, r), node(c, r + 1)))
        elif ry > 0:
            for r in range(ry, 0, -1):
                col_path.append(make_edge(node(c, r), node(c, r - 1)))
        if not col_path:
            continue
        flows.append(
            CollectiveFlow(
                id=f"col{c}",
                label=f"col{c} multicast",
                from_node=node(c, ry),
                to_node=col_path[-1].to_node,
                path=col_path,
                flits=flits,
                release=base,
                color_id=cid,
                payload="broadcast",
            )
        )
        cid += 1
    return flows


def build_tree_reduce_flows(
    rows: int,
    cols: int,
    root: int,
    flits: int = 1,
    payload: str = "reduce",
) -> list[CollectiveFlow]:
    rx, ry = root_coords(root, cols)
    flows: list[CollectiveFlow] = []
    cid = 0
    if ry < rows - 1:
        for c in range(cols):
            for y in range(rows - 1, ry, -1):
                frm, to = node(c, y), node(c, y - 1)
                cid = _make_hop_flow(
                    flows, cid, f"rc{c}y{y}n",
                    f"col{c} ({c},{y})->({c},{y-1})",
                    frm, to, flits, rows - 1 - y, payload,
                )
    if ry > 0:
        for c in range(cols):
            for y in range(ry):
                frm, to = node(c, y), node(c, y + 1)
                cid = _make_hop_flow(
                    flows, cid, f"rc{c}y{y}s",
                    f"col{c} ({c},{y})->({c},{y+1})",
                    frm, to, flits, y, payload,
                )
    row_start = max(ry, rows - 1 - ry)
    if rx < cols - 1:
        for x in range(cols - 1, rx, -1):
            frm, to = node(x, ry), node(x - 1, ry)
            cid = _make_hop_flow(
                flows, cid, f"rr{x}w",
                f"row ({x},{ry})->({x-1},{ry})",
                frm, to, flits, row_start + (cols - 1 - x), payload,
            )
    if rx > 0:
        for x in range(rx):
            frm, to = node(x, ry), node(x + 1, ry)
            cid = _make_hop_flow(
                flows, cid, f"rr{x}e",
                f"row ({x},{ry})->({x+1},{ry})",
                frm, to, flits, row_start + x, payload,
            )
    return flows


def build_gather_to_root_flows(
    rows: int,
    cols: int,
    root: int,
    flits: int = 1,
    payload: str = "gather",
) -> list[CollectiveFlow]:
    rx, ry = root_coords(root, cols)
    root_node = node(rx, ry)
    flows: list[CollectiveFlow] = []
    cid = 0
    for r in range(rows):
        for c in range(cols):
            if r == ry and c == rx:
                continue
            frm = node(c, r)
            path = xy_path(frm, root_node)
            if not path:
                continue
            flows.append(
                CollectiveFlow(
                    id=f"g{r * cols + c}",
                    label=f"({c},{r})->root",
                    from_node=frm,
                    to_node=root_node,
                    path=path,
                    flits=flits,
                    release=0,
                    color_id=cid,
                    payload=payload,
                )
            )
            cid += 1
    return flows


def build_hamilton_gather_flows(
    rows: int,
    cols: int,
    root: int = 0,
    base_flits: int = 1,
) -> tuple[list[CollectiveFlow], int, dict[str, int]]:
    """Hamilton-ring gather to root at cycle[0]; optimal z* = ceil((N-1)/deg(root))."""
    cycle = build_hamilton_cycle(rows, cols)
    n = len(cycle)
    root_node = idx_to_node(root, cols)
    if cycle[0] != root_node:
        # rotate cycle so root is index 0
        pos_i = next(i for i, nd in enumerate(cycle) if nd == root_node)
        cycle = cycle[pos_i:] + cycle[:pos_i]
    t_star = optimal_hamilton_makespan(n)
    pos = {cycle[i]: i for i in range(n)}
    ring_edges = _ring_edges(cycle)
    flows: list[CollectiveFlow] = []
    cid = 0

    for src_idx in range(1, n):
        src_node = cycle[src_idx]
        dist_cw = src_idx
        dist_ccw = n - src_idx
        if dist_ccw <= dist_cw:
            path: list[MeshEdge] = []
            slots: list[int] = []
            for hop in range(dist_ccw):
                ring_pos = (src_idx - hop) % n
                path.append(make_edge(cycle[ring_pos], cycle[(ring_pos - 1) % n]))
                slots.append(hop)
            direction = "ccw"
        else:
            path = []
            slots = []
            for hop in range(dist_cw):
                ring_pos = (src_idx + hop) % n
                path.append(ring_edges[ring_pos])
                slots.append(hop)
            direction = "cw"
        flows.append(
            CollectiveFlow(
                id=f"hg_{direction}_{src_idx}",
                label=f"gather {direction} src={src_idx}",
                from_node=src_node,
                to_node=path[-1].to_node if path else src_node,
                path=path,
                flits=base_flits,
                release=0,
                color_id=cid,
                payload="gather_hamilton",
                slots=slots,
            )
        )
        cid += 1

    meta = {"node_count": n, "t_star": t_star, "flow_count": len(flows)}
    return flows, t_star, meta


def build_dimwise_allreduce_flows(
    rows: int,
    cols: int,
    base_flits: int = 1,
    *,
    pe_switch: bool = True,
) -> list[CollectiveFlow]:
    flows: list[CollectiveFlow] = []
    cid_ref = [0]
    payload = "allreduce"
    release = 0
    for stride in pow2_strides(cols):
        for y in range(rows):
            for x in range(0, cols, 2 * stride):
                for d in range(stride):
                    if x + d + stride < cols:
                        _add_dim_exchange_pair(
                            flows, cid_ref, "x", x + d, y, stride, base_flits,
                            release, payload, "ar", f"X RHD s={stride}", cols,
                        )
        release += dim_exchange_stage_cycles(stride, base_flits)
    if pe_switch and cols > 1 and rows > 1:
        release += PE_INOUT_LATENCY
    for stride in pow2_strides(rows):
        for x in range(cols):
            for y in range(0, rows, 2 * stride):
                for d in range(stride):
                    if y + d + stride < rows:
                        _add_dim_exchange_pair(
                            flows, cid_ref, "y", x, y + d, stride, base_flits,
                            release, payload, "ar", f"Y RHD s={stride}", cols,
                        )
        release += dim_exchange_stage_cycles(stride, base_flits)
    return flows


def build_dimaccum_allreduce_flows(
    rows: int,
    cols: int,
    base_flits: int = 1,
) -> tuple[list[CollectiveFlow], int, dict[str, int]]:
    """Bidirectional dimension accumulation; makespan = X+Y-2 (inline reduce, no PE switch)."""
    flows: list[CollectiveFlow] = []
    cid = 0
    d = mesh_diameter(rows, cols)
    # X sweep: slot t, row y, edge (t,y)->(t+1,y) and (X-2-t,y)->(X-1-t,y)
    for y in range(rows):
        for t in range(cols - 1):
            flows.append(
                CollectiveFlow(
                    id=f"ar_xe_{y}_{t}",
                    label=f"X east row {y} slot {t}",
                    from_node=node(t, y),
                    to_node=node(t + 1, y),
                    path=[make_edge(node(t, y), node(t + 1, y))],
                    flits=base_flits,
                    release=0,
                    color_id=cid,
                    payload="allreduce_dimaccum",
                    slots=[t],
                )
            )
            cid += 1
            wx = cols - 2 - t
            flows.append(
                CollectiveFlow(
                    id=f"ar_xw_{y}_{t}",
                    label=f"X west row {y} slot {t}",
                    from_node=node(wx + 1, y),
                    to_node=node(wx, y),
                    path=[make_edge(node(wx + 1, y), node(wx, y))],
                    flits=base_flits,
                    release=0,
                    color_id=cid,
                    payload="allreduce_dimaccum",
                    slots=[t],
                )
            )
            cid += 1
    x_phase = cols - 1
    for x in range(cols):
        for t in range(rows - 1):
            slot = x_phase + t
            flows.append(
                CollectiveFlow(
                    id=f"ar_ys_{x}_{t}",
                    label=f"Y south col {x} slot {slot}",
                    from_node=node(x, t),
                    to_node=node(x, t + 1),
                    path=[make_edge(node(x, t), node(x, t + 1))],
                    flits=base_flits,
                    release=0,
                    color_id=cid,
                    payload="allreduce_dimaccum",
                    slots=[slot],
                )
            )
            cid += 1
            wy = rows - 2 - t
            flows.append(
                CollectiveFlow(
                    id=f"ar_yn_{x}_{t}",
                    label=f"Y north col {x} slot {slot}",
                    from_node=node(x, wy + 1),
                    to_node=node(x, wy),
                    path=[make_edge(node(x, wy + 1), node(x, wy))],
                    flits=base_flits,
                    release=0,
                    color_id=cid,
                    payload="allreduce_dimaccum",
                    slots=[slot],
                )
            )
            cid += 1
    meta = {"flow_count": len(flows), "t_star": d}
    return flows, d, meta


def build_allto_all_flows(
    rows: int,
    cols: int,
    base_flits: int = 1,
) -> list[CollectiveFlow]:
    n = rows * cols
    flows: list[CollectiveFlow] = []
    cid = 0
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            frm = idx_to_node(s, cols)
            to = idx_to_node(d, cols)
            flows.append(
                CollectiveFlow(
                    id=f"a2a_{s}_{d}",
                    label=f"{s}->{d}",
                    from_node=frm,
                    to_node=to,
                    path=xy_path(frm, to),
                    flits=base_flits,
                    release=0,
                    color_id=cid,
                    payload="alltoall",
                )
            )
            cid += 1
    return flows


def dilate_slots(flows: list[CollectiveFlow], hop_latency: int) -> list[CollectiveFlow]:
    """Stretch a verified h=1 calendar to hop latency h: slot t -> h*t.

    The map t -> h*t is injective, so (edge, slot) uniqueness is preserved;
    consecutive hops t, t+1 map to h*t, h*t+h, satisfying the h-cycle spacing.
    Makespan multiplies by h (launch-slot count; add +h for last-flit arrival).
    """
    for flow in flows:
        if flow.slots is not None:
            flow.slots = [hop_latency * s for s in flow.slots]
    return flows


def verify_hop_spacing(flows: Iterable[CollectiveFlow], hop_latency: int) -> bool:
    """Check every flow's consecutive hop launches are >= hop_latency apart."""
    for flow in flows:
        if flow.slots is None:
            continue
        for j in range(1, len(flow.slots)):
            if flow.slots[j] - flow.slots[j - 1] < hop_latency:
                return False
    return True


def edge_hop_latency(edge: MeshEdge, hx: int, hy: int) -> int:
    """Latency of one edge: hx for horizontal (E/W), hy for vertical (N/S)."""
    return hx if edge.from_node.y == edge.to_node.y else hy


def verify_hop_spacing_aniso(
    flows: Iterable[CollectiveFlow],
    hx: int,
    hy: int,
) -> bool:
    """Per-edge spacing: next hop launches >= latency of the edge just crossed."""
    for flow in flows:
        if flow.slots is None:
            continue
        for j in range(1, len(flow.slots)):
            need = edge_hop_latency(flow.path[j - 1], hx, hy)
            if flow.slots[j] - flow.slots[j - 1] < need:
                return False
    return True


def weighted_eccentricity(rows: int, cols: int, root: int, hx: int, hy: int) -> int:
    rx, ry = root_coords(root, cols)
    farthest = 0
    for y in range(rows):
        for x in range(cols):
            farthest = max(farthest, hx * abs(x - rx) + hy * abs(y - ry))
    return farthest


def weighted_diameter(rows: int, cols: int, hx: int, hy: int) -> int:
    return hx * (cols - 1) + hy * (rows - 1)


def build_alltoall_twophase_flows(
    rows: int,
    cols: int,
    base_flits: int = 1,
    hop_latency: int = 1,
    hop_latency_x: int | None = None,
    hop_latency_y: int | None = None,
) -> list[CollectiveFlow]:
    """All-to-all as a preassigned, conflict-free (stall=0) calendar.

    Two-phase dimension-ordered construction (report Section 5.1):
      * Phase X: every (s,d) message moves along its row (sx -> dx). Row links
        are first-fit interval-colored, so each directed row edge carries <= 1
        flit per slot. Takes Tx slots.
      * PE relay: the dimension turn at (dx, sy) ejects to PE and re-injects on
        the column, separated by a global barrier B = Tx + PE_INOUT_LATENCY.
      * Phase Y: every message moves along its column (sy -> dy), column links
        first-fit interval-colored.

    Each flow keeps its XY path; slots are contiguous within a phase with a gap
    at the turn (the planned PE relay). Result: peak=1, stall=0 (no online
    source contention), makespan Z ~= N*cols/4 + N*rows/4 <= 2*z*.
    """
    n = rows * cols
    hx = max(1, hop_latency_x if hop_latency_x is not None else hop_latency)
    hy = max(1, hop_latency_y if hop_latency_y is not None else hop_latency)
    reserved: dict[str, set[int]] = {}

    def first_free(edges: list[MeshEdge], stride: int, start: int) -> int:
        # Successive hops of one flit launch `stride` cycles apart (the axis
        # latency); each link still accepts one new flit per cycle (II=1).
        phi = start
        while True:
            if all(
                phi + j * stride not in reserved.get(e.id, ())
                for j, e in enumerate(edges)
            ):
                return phi
            phi += 1

    def occupy(edges: list[MeshEdge], stride: int, phi: int) -> None:
        for j, e in enumerate(edges):
            reserved.setdefault(e.id, set()).add(phi + j * stride)

    items = []
    for s in range(n):
        sx, sy = s % cols, s // cols
        for d in range(n):
            if s == d:
                continue
            dx, dy = d % cols, d // cols
            path = xy_path(idx_to_node(s, cols), idx_to_node(d, cols))
            nx = abs(dx - sx)
            items.append((s, d, sx, sy, dx, dy, path, nx))

    # Phase X: row moves, first-fit by left column endpoint (optimal interval coloring).
    x_slots: dict[tuple[int, int], list[int]] = {}
    tx = 0
    x_items = [it for it in items if it[7] > 0]
    x_items.sort(key=lambda it: (min(it[2], it[4]), abs(it[4] - it[2])))
    for s, d, sx, sy, dx, dy, path, nx in x_items:
        x_edges = path[:nx]
        phi = first_free(x_edges, hx, 0)
        occupy(x_edges, hx, phi)
        x_slots[(s, d)] = [phi + j * hx for j in range(nx)]
        tx = max(tx, phi + nx * hx)  # arrival of the last X-phase flit

    barrier = tx + PE_INOUT_LATENCY

    # Phase Y: column moves, injected at/after the barrier.
    y_slots: dict[tuple[int, int], list[int]] = {}
    y_items = [it for it in items if len(it[6]) - it[7] > 0]
    y_items.sort(key=lambda it: (min(it[3], it[5]), abs(it[5] - it[3])))
    for s, d, sx, sy, dx, dy, path, nx in y_items:
        y_edges = path[nx:]
        phi = first_free(y_edges, hy, barrier)
        occupy(y_edges, hy, phi)
        y_slots[(s, d)] = [phi + j * hy for j in range(len(y_edges))]

    flows: list[CollectiveFlow] = []
    cid = 0
    for s, d, sx, sy, dx, dy, path, nx in items:
        slots: list[int] = []
        if nx > 0:
            slots += x_slots[(s, d)]
        if len(path) - nx > 0:
            slots += y_slots[(s, d)]
        flows.append(
            CollectiveFlow(
                id=f"a2a2p_{s}_{d}",
                label=f"{s}->{d}",
                from_node=idx_to_node(s, cols),
                to_node=idx_to_node(d, cols),
                path=path,
                flits=base_flits,
                release=0,
                color_id=cid,
                payload="alltoall_twophase",
                slots=slots,
            )
        )
        cid += 1
    return flows


def assign_rigid_slots(flows: list[CollectiveFlow]) -> None:
    """release + hop rigid placement for edge-disjoint tree flows."""
    for flow in flows:
        flow.slots = [flow.release + hop for hop in range(len(flow.path))]


def max_router_calendar_depth(
    flows: Iterable[CollectiveFlow],
    rows: int,
    cols: int,
) -> int:
    """Max per-router span of active slots (one-shot calendar table depth)."""
    per_node: dict[int, list[int]] = {}
    for flow in flows:
        if not flow.slots or not flow.path:
            continue
        for hop, edge in enumerate(flow.path):
            slot = flow.slots[hop]
            for endpoint in (edge.from_node, edge.to_node):
                idx = node_idx(endpoint, cols)
                per_node.setdefault(idx, []).append(slot)
    if not per_node:
        return 0
    return max(max(slots) - min(slots) + 1 for slots in per_node.values())


def analyze_collective(
    pattern: str,
    rows: int,
    cols: int,
    *,
    root: int = 0,
    variant: str | None = None,
) -> dict:
    """Analyze one collective: z lower bound, scheduled makespan, L*, router table sizes."""
    pattern = pattern.lower()
    n = rows * cols
    z_lb = optimal_z_lower_bound(pattern, rows, cols, root=root)

    if pattern == "broadcast":
        flows = build_multicast_broadcast_flows(rows, cols, root)
        assign_rigid_slots(flows)
        sched = verify_preassigned_slots(flows)
        method = "multicast tree (row + column buses)"
    elif pattern == "reduce":
        flows = build_tree_reduce_flows(rows, cols, root)
        assign_rigid_slots(flows)
        sched = verify_preassigned_slots(flows)
        method = "reverse multicast tree + inline reduce"
    elif pattern == "allreduce":
        flows, z_opt, _ = build_dimaccum_allreduce_flows(rows, cols)
        sched = verify_preassigned_slots(flows)
        method = "bidirectional dimension accumulation"
        z_lb = z_opt
    elif pattern == "allgather":
        flows, z_opt, _ = build_hamilton_ring_allgather_flows(rows, cols)
        sched = verify_preassigned_slots(flows)
        method = "Hamilton bidirectional ring pipeline"
        z_lb = z_opt
    elif pattern == "gather":
        flows, z_opt, _ = build_hamilton_gather_flows(rows, cols, root)
        sched = verify_preassigned_slots(flows)
        method = "Hamilton ring feed to root"
        z_lb = min(z_lb, z_opt)
    elif pattern == "alltoall":
        if variant == "twophase":
            flows = build_alltoall_twophase_flows(rows, cols)
            sched = verify_preassigned_slots(flows)
            method = "2-phase dim-ordered + interval coloring (preassigned, stall=0)"
        else:
            flows = build_allto_all_flows(rows, cols)
            sched = schedule_bufferless_noc(flows)
            method = "XY unicast (bufferless greedy schedule)"
    else:
        raise ValueError(pattern)

    load = compute_edge_load(flows)
    router_depth = max_router_calendar_depth(flows, rows, cols) if sched.ok else 0
    if sched.ok and sched.makespan:
        router_table_oneshot = sched.makespan
    else:
        router_table_oneshot = sched.makespan

    return {
        "pattern": pattern,
        "rows": rows,
        "cols": cols,
        "node_count": n,
        "root": root,
        "method": method,
        "z_lower_bound": z_lb,
        "z_scheduled": sched.makespan,
        "stall": sched.total_stall,
        "ok": sched.ok,
        "peak": sched.peak,
        "l_star": load.l_star,
        "p_min_periodic": load.l_star,
        "router_table_oneshot": router_table_oneshot,
        "router_table_periodic": load.l_star,
        "router_max_active_span": router_depth,
        "flow_count": len(flows),
        "witness_edge": load.witness,
    }


def analyze_all_collectives(
    meshes: list[tuple[int, int]],
    *,
    root: int = 0,
) -> list[dict]:
    patterns = ("broadcast", "reduce", "allreduce", "allgather", "gather", "alltoall")
    out: list[dict] = []
    for rows, cols in meshes:
        for pattern in patterns:
            out.append(analyze_collective(pattern, rows, cols, root=root))
        a2a_2p = analyze_collective("alltoall", rows, cols, root=root, variant="twophase")
        a2a_2p["pattern"] = "alltoall_twophase"
        out.append(a2a_2p)
    return out

"""Fault-tolerance analysis for mesh collectives under PE/link defects."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Literal

from wsesim.network.collective_patterns import (
    CollectiveFlow,
    MeshEdge,
    MeshNode,
    analyze_collective,
    assign_rigid_slots,
    build_allto_all_flows,
    build_dimaccum_allreduce_flows,
    build_hamilton_cycle,
    build_hamilton_gather_flows,
    build_hamilton_ring_allgather_flows,
    build_multicast_broadcast_flows,
    build_tree_reduce_flows,
    eid,
    idx_to_node,
    make_edge,
    node,
    node_idx,
    optimal_hamilton_makespan,
    root_coords,
    schedule_bufferless_noc,
    verify_preassigned_slots,
    xy_path,
)

FaultType = Literal["pe_point", "link_point", "pe_block", "pe_block_1x2"]
Region = Literal["corner", "edge", "center"]


@dataclass(frozen=True, slots=True)
class MeshFault:
    bad_nodes: frozenset[tuple[int, int]] = frozenset()
    bad_edges: frozenset[str] = frozenset()

    @classmethod
    def pe(cls, x: int, y: int) -> MeshFault:
        return cls(bad_nodes=frozenset({(x, y)}))

    @classmethod
    def link(cls, a: MeshNode, b: MeshNode) -> MeshFault:
        return cls(bad_edges=frozenset({eid(a, b), eid(b, a)}))

    @classmethod
    def block(cls, x: int, y: int) -> MeshFault:
        nodes = frozenset((x + dx, y + dy) for dx in (0, 1) for dy in (0, 1))
        return cls(bad_nodes=nodes)

    @classmethod
    def block1x2(cls, x: int, y: int, *, horizontal: bool = True) -> MeshFault:
        b = (x + 1, y) if horizontal else (x, y + 1)
        return cls(bad_nodes=frozenset({(x, y), b}))

    def node_bad(self, n: MeshNode) -> bool:
        return (n.x, n.y) in self.bad_nodes

    def edge_bad(self, edge: MeshEdge) -> bool:
        return edge.id in self.bad_edges

    def path_valid(self, path: Iterable[MeshEdge]) -> bool:
        for edge in path:
            if self.edge_bad(edge):
                return False
            if self.node_bad(edge.from_node) or self.node_bad(edge.to_node):
                return False
        return True


def node_region(rows: int, cols: int, x: int, y: int) -> Region:
    on_left = x == 0
    on_right = x == cols - 1
    on_top = y == 0
    on_bottom = y == rows - 1
    if (on_left or on_right) and (on_top or on_bottom):
        return "corner"
    if on_left or on_right or on_top or on_bottom:
        return "edge"
    return "center"


def region_point(rows: int, cols: int, region: Region, *, root: int = 0) -> tuple[int, int]:
    rx, ry = root_coords(root, cols)
    if region == "corner":
        return rx, ry
    if region == "edge":
        return cols // 2, 0
    return cols // 2, rows // 2


def region_block_origin(rows: int, cols: int, region: Region) -> tuple[int, int]:
    if region == "corner":
        return 0, 0
    if region == "edge":
        return max(0, cols // 2 - 1), 0
    return max(0, cols // 2 - 1), max(0, rows // 2 - 1)


def region_block1x2(
    rows: int, cols: int, region: Region
) -> tuple[int, int, bool]:
    """Origin (x, y) and orientation (horizontal) of a 1×2 PE block per region."""
    if region == "corner":
        return 0, 0, True
    if region == "edge":
        return cols // 2, 0, True
    return cols // 2 - 1, rows // 2, True


def make_scenario_fault(
    rows: int,
    cols: int,
    fault_type: FaultType,
    region: Region,
    *,
    root: int = 0,
) -> MeshFault:
    if fault_type == "pe_block":
        x, y = region_block_origin(rows, cols, region)
        if x + 1 >= cols or y + 1 >= rows:
            x, y = region_point(rows, cols, region, root=root)
            return MeshFault.pe(x, y)
        return MeshFault.block(x, y)
    if fault_type == "pe_block_1x2":
        x, y, horizontal = region_block1x2(rows, cols, region)
        bx, by = (x + 1, y) if horizontal else (x, y + 1)
        if bx >= cols or by >= rows:
            return MeshFault.pe(x, y)
        return MeshFault.block1x2(x, y, horizontal=horizontal)
    x, y = region_point(rows, cols, region, root=root)
    if fault_type == "pe_point":
        return MeshFault.pe(x, y)
    if region == "corner":
        a, b = node(x, y), node(x + 1, y)
    elif region == "edge":
        a, b = node(x, y), node(x, y + 1)
    else:
        a, b = node(x, y), node(x + 1, y)
    return MeshFault.link(a, b)


def survivor_nodes(rows: int, cols: int, fault: MeshFault) -> list[MeshNode]:
    return [
        node(x, y)
        for y in range(rows)
        for x in range(cols)
        if (x, y) not in fault.bad_nodes
    ]


def _neighbors(n: MeshNode, rows: int, cols: int) -> list[MeshNode]:
    out: list[MeshNode] = []
    if n.x > 0:
        out.append(node(n.x - 1, n.y))
    if n.x < cols - 1:
        out.append(node(n.x + 1, n.y))
    if n.y > 0:
        out.append(node(n.x, n.y - 1))
    if n.y < rows - 1:
        out.append(node(n.x, n.y + 1))
    return out


def bfs_path(
    start: MeshNode,
    goal: MeshNode,
    rows: int,
    cols: int,
    fault: MeshFault,
) -> list[MeshEdge] | None:
    if fault.node_bad(start) or fault.node_bad(goal):
        return None
    if start.x == goal.x and start.y == goal.y:
        return []
    prev: dict[MeshNode, tuple[MeshNode, MeshEdge] | None] = {start: None}
    q: deque[MeshNode] = deque([start])
    while q:
        cur = q.popleft()
        for nxt in _neighbors(cur, rows, cols):
            if fault.node_bad(nxt) or nxt in prev:
                continue
            edge = make_edge(cur, nxt)
            if fault.edge_bad(edge):
                continue
            prev[nxt] = (cur, edge)
            if nxt.x == goal.x and nxt.y == goal.y:
                path: list[MeshEdge] = []
                walk: MeshNode | None = nxt
                while walk is not None and prev[walk] is not None:
                    parent, e = prev[walk]
                    path.append(e)
                    walk = parent
                path.reverse()
                return path
            q.append(nxt)
    return None


def reroute_flow(
    flow: CollectiveFlow,
    rows: int,
    cols: int,
    fault: MeshFault,
) -> CollectiveFlow | None:
    if fault.node_bad(flow.from_node) or fault.node_bad(flow.to_node):
        return None
    if fault.path_valid(flow.path):
        return flow
    new_path = bfs_path(flow.from_node, flow.to_node, rows, cols, fault)
    if new_path is None:
        return None
    new_flow = CollectiveFlow(
        id=flow.id,
        from_node=flow.from_node,
        to_node=flow.to_node,
        path=new_path,
        flits=flow.flits,
        release=flow.release,
        color_id=flow.color_id,
        payload=flow.payload,
        label=flow.label,
        slots=None,
    )
    return new_flow


def reroute_flows(
    flows: list[CollectiveFlow],
    rows: int,
    cols: int,
    fault: MeshFault,
) -> tuple[list[CollectiveFlow], int]:
    """Reroute flows around faults; drop unreachable. Returns (flows, dropped)."""
    out: list[CollectiveFlow] = []
    dropped = 0
    for flow in flows:
        routed = reroute_flow(flow, rows, cols, fault)
        if routed is None:
            dropped += 1
        else:
            out.append(routed)
    return out, dropped


def bipartite_survivor_counts(
    rows: int,
    cols: int,
    fault: MeshFault,
) -> tuple[int, int]:
    even = odd = 0
    for nd in survivor_nodes(rows, cols, fault):
        if (nd.x + nd.y) % 2 == 0:
            even += 1
        else:
            odd += 1
    return even, odd


def hamilton_cycle_possible(rows: int, cols: int, fault: MeshFault) -> bool:
    even, odd = bipartite_survivor_counts(rows, cols, fault)
    return even == odd and even > 0


def _survivor_snake_order(rows: int, cols: int, fault: MeshFault) -> list[MeshNode]:
    order: list[MeshNode] = []
    for y in range(rows):
        xs = range(cols) if y % 2 == 0 else range(cols - 1, -1, -1)
        for x in xs:
            if (x, y) not in fault.bad_nodes:
                order.append(node(x, y))
    return order


def _is_adjacent(a: MeshNode, b: MeshNode) -> bool:
    return abs(a.x - b.x) + abs(a.y - b.y) == 1


def _warnsdorff_degree(
    n: MeshNode,
    rows: int,
    cols: int,
    fault: MeshFault,
    visited: set[tuple[int, int]],
) -> int:
    count = 0
    for nn in _neighbors(n, rows, cols):
        if (nn.x, nn.y) in fault.bad_nodes or (nn.x, nn.y) in visited:
            continue
        count += 1
    return count


def _hamilton_path_backtrack(
    rows: int,
    cols: int,
    fault: MeshFault,
    start: MeshNode,
) -> list[MeshNode] | None:
    survivors = survivor_nodes(rows, cols, fault)
    target = len(survivors)
    visited: set[tuple[int, int]] = {(start.x, start.y)}
    path = [start]

    def dfs(cur: MeshNode) -> bool:
        if len(path) == target:
            return True
        cands = [
            nxt
            for nxt in _neighbors(cur, rows, cols)
            if (nxt.x, nxt.y) not in fault.bad_nodes
            and (nxt.x, nxt.y) not in visited
        ]
        cands.sort(
            key=lambda n: (
                _warnsdorff_degree(n, rows, cols, fault, visited),
                n.x,
                n.y,
            )
        )
        for nxt in cands:
            path.append(nxt)
            visited.add((nxt.x, nxt.y))
            if dfs(nxt):
                return True
            path.pop()
            visited.remove((nxt.x, nxt.y))
        return False

    if dfs(start):
        return path
    return None


def _hamilton_path_multi_start(
    rows: int,
    cols: int,
    fault: MeshFault,
    preferred: MeshNode | None = None,
) -> list[MeshNode]:
    survivors = survivor_nodes(rows, cols, fault)
    if not survivors:
        return []
    starts = [preferred] if preferred and not fault.node_bad(preferred) else []
    for nd in survivors:
        if nd not in starts:
            starts.append(nd)
    for start in starts:
        path = _hamilton_path_backtrack(rows, cols, fault, start)
        if path:
            return path
    return []


def _stitch_snake_with_detours(
    rows: int,
    cols: int,
    fault: MeshFault,
    snake: list[MeshNode],
) -> list[MeshNode]:
    """Turn a snake order with gaps into a walk by BFS detours between breaks."""
    if len(snake) <= 1:
        return snake
    order = [snake[0]]
    visited = {(snake[0].x, snake[0].y)}
    for i in range(1, len(snake)):
        prev = order[-1]
        nxt = snake[i]
        if _is_adjacent(prev, nxt):
            order.append(nxt)
            visited.add((nxt.x, nxt.y))
            continue
        bridge = bfs_path(prev, nxt, rows, cols, fault)
        if bridge is None:
            return []
        for edge in bridge:
            nd = edge.to_node
            if (nd.x, nd.y) in visited and nd != nxt:
                return []
            if (nd.x, nd.y) not in visited:
                order.append(nd)
                visited.add((nd.x, nd.y))
    return order


def build_hamilton_on_punctured(
    rows: int,
    cols: int,
    fault: MeshFault,
    *,
    start: MeshNode | None = None,
) -> tuple[list[MeshNode], bool]:
    """Return (visit order, is_cycle) on survivor mesh."""
    if not fault.bad_nodes and not fault.bad_edges:
        cycle = build_hamilton_cycle(rows, cols)
        return cycle, True

    survivors = survivor_nodes(rows, cols, fault)
    if not survivors:
        return [], False

    if start is None:
        start = survivors[0]
    elif fault.node_bad(start):
        start = survivors[0]

    snake = _survivor_snake_order(rows, cols, fault)
    snake_ok = len(snake) > 1 and all(
        _is_adjacent(snake[i], snake[i + 1]) for i in range(len(snake) - 1)
    )
    if snake_ok:
        order = snake
    elif len(survivors) <= 20:
        order = _hamilton_path_multi_start(rows, cols, fault, preferred=start)
    else:
        order = _stitch_snake_with_detours(rows, cols, fault, snake)

    if not order:
        return [], False

    is_cycle = (
        len(order) > 2
        and _is_adjacent(order[0], order[-1])
        and hamilton_cycle_possible(rows, cols, fault)
        and len(order) == len(survivors)
    )
    return order, is_cycle


def _ring_edges_from_order(order: list[MeshNode], is_cycle: bool) -> list[MeshEdge]:
    edges: list[MeshEdge] = []
    for i in range(len(order) - 1):
        edges.append(make_edge(order[i], order[i + 1]))
    if is_cycle and len(order) > 1:
        edges.append(make_edge(order[-1], order[0]))
    return edges


def build_faulty_hamilton_allgather_flows(
    rows: int,
    cols: int,
    fault: MeshFault,
    base_flits: int = 1,
) -> tuple[list[CollectiveFlow], int, dict]:
    order, is_cycle = build_hamilton_on_punctured(rows, cols, fault)
    if not order:
        return [], 0, {"reachable": False, "is_cycle": False}

    n = len(order)
    pos = {order[i]: i for i in range(n)}
    ring_edges = _ring_edges_from_order(order, is_cycle)
    if is_cycle:
        t_star = optimal_hamilton_makespan(n)
    else:
        t_star = n - 1

    flows: list[CollectiveFlow] = []
    cid = 0

    for src_idx, src_node in enumerate(order):
        pos_src = pos[src_node]
        if is_cycle:
            directions = ("cw", "ccw")
        else:
            directions = ("cw",)

        for direction in directions:
            path: list[MeshEdge] = []
            slots: list[int] = []
            for hop in range(t_star):
                if direction == "cw":
                    if is_cycle:
                        edge = ring_edges[(pos_src + hop) % len(ring_edges)]
                    else:
                        idx = pos_src + hop
                        if idx + 1 >= len(order):
                            break
                        edge = make_edge(order[idx], order[idx + 1])
                else:
                    ring_pos = (pos_src - hop) % n
                    edge = make_edge(order[ring_pos], order[(ring_pos - 1) % n])
                if fault.edge_bad(edge) or fault.node_bad(edge.from_node) or fault.node_bad(edge.to_node):
                    detour = bfs_path(
                        path[-1].to_node if path else src_node,
                        edge.to_node,
                        rows,
                        cols,
                        fault,
                    )
                    if detour is None:
                        path = []
                        break
                    for j, e in enumerate(detour):
                        path.append(e)
                        slots.append(hop if j == 0 else slots[-1] + 1)
                    continue
                path.append(edge)
                slots.append(hop)
            if not path:
                continue
            flows.append(
                CollectiveFlow(
                    id=f"fhra_{direction}_{src_idx}",
                    label=f"faulty hr {direction} src={src_idx}",
                    from_node=src_node,
                    to_node=path[-1].to_node,
                    path=path,
                    flits=base_flits,
                    release=0,
                    color_id=cid,
                    payload="allgather_faulty",
                    slots=slots if len(slots) == len(path) else None,
                )
            )
            cid += 1

    meta = {
        "node_count": n,
        "t_star": t_star,
        "is_cycle": is_cycle,
        "flow_count": len(flows),
        "reachable": len(flows) > 0,
    }
    return flows, t_star, meta


def build_faulty_hamilton_gather_flows(
    rows: int,
    cols: int,
    root: int,
    fault: MeshFault,
    base_flits: int = 1,
) -> tuple[list[CollectiveFlow], int, dict]:
    root_node = idx_to_node(root, cols)
    order, is_cycle = build_hamilton_on_punctured(rows, cols, fault, start=root_node)
    if not order or fault.node_bad(root_node):
        return [], 0, {"reachable": False, "is_cycle": False}

    if order[0] != root_node:
        pos_i = next(i for i, nd in enumerate(order) if nd == root_node)
        order = order[pos_i:] + order[:pos_i]

    n = len(order)
    pos = {order[i]: i for i in range(n)}
    ring_edges = _ring_edges_from_order(order, is_cycle)
    t_star = optimal_hamilton_makespan(n) if is_cycle else n - 1

    flows: list[CollectiveFlow] = []
    cid = 0

    for src_idx in range(1, n):
        src_node = order[src_idx]
        dist_cw = src_idx
        dist_ccw = n - src_idx if is_cycle else dist_cw + 1
        use_ccw = is_cycle and dist_ccw <= dist_cw

        path: list[MeshEdge] = []
        slots: list[int] = []
        hops = dist_ccw if use_ccw else dist_cw
        for hop in range(hops):
            if use_ccw:
                ring_pos = (src_idx - hop) % n
                edge = make_edge(order[ring_pos], order[(ring_pos - 1) % n])
            elif is_cycle:
                edge = ring_edges[(pos[src_node] + hop) % len(ring_edges)]
            else:
                idx = src_idx + hop
                if idx + 1 >= len(order):
                    break
                edge = make_edge(order[idx], order[idx + 1])
            if fault.edge_bad(edge) or fault.node_bad(edge.from_node) or fault.node_bad(edge.to_node):
                detour = bfs_path(
                    path[-1].to_node if path else src_node,
                    edge.to_node,
                    rows,
                    cols,
                    fault,
                )
                if detour is None:
                    path = []
                    break
                path.extend(detour)
                slots.extend(range(len(slots), len(slots) + len(detour)))
                continue
            path.append(edge)
            slots.append(hop)

        if not path:
            continue
        flows.append(
            CollectiveFlow(
                id=f"fhg_{src_idx}",
                label=f"faulty gather src={src_idx}",
                from_node=src_node,
                to_node=path[-1].to_node,
                path=path,
                flits=base_flits,
                release=0,
                color_id=cid,
                payload="gather_faulty",
                slots=slots if len(slots) == len(path) else None,
            )
        )
        cid += 1

    meta = {
        "node_count": n,
        "t_star": t_star,
        "is_cycle": is_cycle,
        "flow_count": len(flows),
        "reachable": len(flows) > 0,
    }
    return flows, t_star, meta


def build_faulty_broadcast_flows(
    rows: int,
    cols: int,
    root: int,
    fault: MeshFault,
    flits: int = 1,
) -> tuple[list[CollectiveFlow], int]:
    if fault.node_bad(idx_to_node(root, cols)):
        return [], 0
    flows = build_multicast_broadcast_flows(rows, cols, root, flits)
    routed, _ = reroute_flows(flows, rows, cols, fault)
    return routed, len(routed)


def build_faulty_reduce_flows(
    rows: int,
    cols: int,
    root: int,
    fault: MeshFault,
    flits: int = 1,
) -> tuple[list[CollectiveFlow], int]:
    if fault.node_bad(idx_to_node(root, cols)):
        return [], 0
    flows = build_tree_reduce_flows(rows, cols, root, flits)
    routed, dropped = reroute_flows(flows, rows, cols, fault)
    survivors = len(survivor_nodes(rows, cols, fault)) - 1
    return routed, max(0, survivors - dropped)


def build_faulty_allreduce_flows(
    rows: int,
    cols: int,
    fault: MeshFault,
    base_flits: int = 1,
) -> tuple[list[CollectiveFlow], int]:
    flows, z_opt, _ = build_dimaccum_allreduce_flows(rows, cols, base_flits)
    routed, _ = reroute_flows(flows, rows, cols, fault)
    return routed, z_opt


def build_faulty_alltoall_flows(
    rows: int,
    cols: int,
    fault: MeshFault,
    base_flits: int = 1,
) -> tuple[list[CollectiveFlow], int]:
    flows = build_allto_all_flows(rows, cols, base_flits)
    routed, dropped = reroute_flows(flows, rows, cols, fault)
    n = rows * cols
    expected = n * (n - 1) - dropped
    return routed, expected


def _build_healthy_flows_for_schedule(
    pattern: str,
    rows: int,
    cols: int,
    *,
    root: int = 0,
) -> list[CollectiveFlow]:
    pattern = pattern.lower()
    if pattern == "broadcast":
        flows = build_multicast_broadcast_flows(rows, cols, root)
        assign_rigid_slots(flows)
    elif pattern == "reduce":
        flows = build_tree_reduce_flows(rows, cols, root)
        assign_rigid_slots(flows)
    elif pattern == "allreduce":
        flows, _, _ = build_dimaccum_allreduce_flows(rows, cols)
    elif pattern == "allgather":
        flows, _, _ = build_hamilton_ring_allgather_flows(rows, cols)
    elif pattern == "gather":
        flows, _, _ = build_hamilton_gather_flows(rows, cols, root)
    elif pattern == "alltoall":
        flows = build_allto_all_flows(rows, cols)
    else:
        raise ValueError(pattern)
    return flows


def _build_faulty_flows(
    pattern: str,
    rows: int,
    cols: int,
    fault: MeshFault,
    *,
    root: int = 0,
) -> tuple[list[CollectiveFlow], dict]:
    pattern = pattern.lower()
    meta: dict = {"hamilton_degraded": False, "is_cycle": True, "reachable": True}

    if pattern == "broadcast":
        flows, _ = build_faulty_broadcast_flows(rows, cols, root, fault)
        assign_rigid_slots(flows)
    elif pattern == "reduce":
        flows, _ = build_faulty_reduce_flows(rows, cols, root, fault)
        assign_rigid_slots(flows)
    elif pattern == "allreduce":
        flows, _ = build_faulty_allreduce_flows(rows, cols, fault)
    elif pattern == "allgather":
        flows, _, ham_meta = build_faulty_hamilton_allgather_flows(rows, cols, fault)
        meta.update(ham_meta)
        meta["hamilton_degraded"] = not ham_meta.get("is_cycle", True)
    elif pattern == "gather":
        flows, _, ham_meta = build_faulty_hamilton_gather_flows(rows, cols, root, fault)
        meta.update(ham_meta)
        meta["hamilton_degraded"] = not ham_meta.get("is_cycle", True)
    elif pattern == "alltoall":
        flows, _ = build_faulty_alltoall_flows(rows, cols, fault)
    else:
        raise ValueError(pattern)

    if not flows:
        meta["reachable"] = False
    return flows, meta


def schedule_collective_flows(flows: list[CollectiveFlow]) -> dict:
    """Schedule flows; prefer preassigned slots when complete."""
    if flows and all(
        f.slots is not None and len(f.slots) == len(f.path) for f in flows if f.path
    ):
        sched = verify_preassigned_slots(flows)
    else:
        for f in flows:
            f.slots = None
        sched = schedule_bufferless_noc(flows)
    return {
        "ok": sched.ok,
        "makespan": sched.makespan,
        "stall": sched.total_stall,
        "peak": sched.peak,
        "collision": sched.collision,
    }


def schedule_collective_fair(flows: list[CollectiveFlow]) -> dict:
    """Bufferless schedule on fresh flow copies (fair healthy vs faulty compare)."""
    copies = [
        CollectiveFlow(
            id=f.id,
            from_node=f.from_node,
            to_node=f.to_node,
            path=list(f.path),
            flits=f.flits,
            release=f.release,
            color_id=f.color_id,
            payload=f.payload,
            label=f.label,
            slots=None,
        )
        for f in flows
    ]
    sched = schedule_bufferless_noc(copies)
    return {
        "ok": sched.ok,
        "makespan": sched.makespan,
        "stall": sched.total_stall,
        "peak": sched.peak,
        "collision": sched.collision,
    }


def analyze_collective_faulty(
    pattern: str,
    rows: int,
    cols: int,
    fault: MeshFault,
    *,
    root: int = 0,
    region: str = "",
    fault_type: str = "",
) -> dict:
    healthy_ref = analyze_collective(pattern, rows, cols, root=root)
    healthy_flows = _build_healthy_flows_for_schedule(pattern, rows, cols, root=root)
    healthy_sched = schedule_collective_fair(healthy_flows)

    faulty_flows, meta = _build_faulty_flows(pattern, rows, cols, fault, root=root)
    faulty_sched = schedule_collective_fair(faulty_flows)

    z_healthy = healthy_sched["makespan"] or healthy_ref["z_scheduled"]
    z_faulty = faulty_sched["makespan"]

    n = rows * cols
    full_flow_counts = {
        "broadcast": len(healthy_flows),
        "reduce": n - 1,
        "allreduce": len(healthy_flows),
        "allgather": len(healthy_flows),
        "gather": n - 1,
        "alltoall": n * (n - 1),
    }
    expected_full = full_flow_counts.get(pattern.lower(), len(healthy_flows))
    flow_count_faulty = len(faulty_flows)
    partial = flow_count_faulty < expected_full

    if z_healthy and z_faulty and not partial:
        ratio = round(z_faulty / z_healthy, 3)
    elif z_healthy and z_faulty and partial:
        ratio = round(max(1.0, z_faulty / z_healthy), 3)
    else:
        ratio = None

    note_parts: list[str] = []
    if not meta.get("reachable", True):
        note_parts.append("不可达/无流")
    if partial:
        note_parts.append(f"部分流({flow_count_faulty}/{expected_full})")
    if meta.get("hamilton_degraded"):
        note_parts.append("哈密顿环→路径")
    if not faulty_sched["ok"]:
        note_parts.append(f"调度冲突:{faulty_sched.get('collision')}")

    return {
        "pattern": pattern.lower(),
        "rows": rows,
        "cols": cols,
        "mesh": f"{rows}×{cols}",
        "root": root,
        "region": region,
        "fault_type": fault_type,
        "bad_nodes": sorted(fault.bad_nodes),
        "bad_edges": sorted(fault.bad_edges),
        "z_healthy_ref": healthy_ref["z_scheduled"],
        "z_healthy_scheduled": z_healthy,
        "z_faulty": z_faulty,
        "ratio": round(ratio, 3) if ratio is not None else None,
        "stall_healthy": healthy_sched["stall"],
        "stall_faulty": faulty_sched["stall"],
        "peak_healthy": healthy_sched["peak"],
        "peak_faulty": faulty_sched["peak"],
        "ok_healthy": healthy_sched["ok"],
        "ok_faulty": faulty_sched["ok"],
        "reachable": meta.get("reachable", True),
        "hamilton_degraded": meta.get("hamilton_degraded", False),
        "is_cycle": meta.get("is_cycle", True),
        "flow_count_faulty": flow_count_faulty,
        "flow_count_expected": expected_full,
        "partial_coverage": partial,
        "note": "; ".join(note_parts) if note_parts else "",
    }


FAULT_TYPES: tuple[FaultType, ...] = ("pe_point", "link_point", "pe_block")
REGIONS: tuple[Region, ...] = ("corner", "edge", "center")
PATTERNS: tuple[str, ...] = (
    "broadcast",
    "reduce",
    "allreduce",
    "allgather",
    "gather",
    "alltoall",
)


def analyze_fault_matrix(
    meshes: list[tuple[int, int]],
    *,
    root: int = 0,
    fault_types: Iterable[FaultType] = FAULT_TYPES,
    regions: Iterable[Region] = REGIONS,
    patterns: Iterable[str] = PATTERNS,
) -> list[dict]:
    results: list[dict] = []
    healthy_cache: dict[tuple[str, int, int], dict] = {}

    for rows, cols in meshes:
        for pattern in patterns:
            key = (pattern.lower(), rows, cols)
            if key not in healthy_cache:
                healthy_ref = analyze_collective(pattern, rows, cols, root=root)
                healthy_flows = _build_healthy_flows_for_schedule(
                    pattern, rows, cols, root=root
                )
                healthy_sched = schedule_collective_fair(healthy_flows)
                healthy_cache[key] = {
                    "ref": healthy_ref,
                    "flows": healthy_flows,
                    "sched": healthy_sched,
                }

        for fault_type in fault_types:
            for region in regions:
                fault = make_scenario_fault(
                    rows, cols, fault_type, region, root=root
                )
                for pattern in patterns:
                    key = (pattern.lower(), rows, cols)
                    cached = healthy_cache[key]
                    row = _analyze_with_cached_healthy(
                        pattern,
                        rows,
                        cols,
                        fault,
                        cached,
                        root=root,
                        region=region,
                        fault_type=fault_type,
                    )
                    results.append(row)
                    print(
                        f"fault {rows}x{cols} {pattern} {fault_type} {region} "
                        f"ratio={row.get('ratio')}",
                        flush=True,
                    )
    return results


def _analyze_with_cached_healthy(
    pattern: str,
    rows: int,
    cols: int,
    fault: MeshFault,
    healthy: dict,
    *,
    root: int = 0,
    region: str = "",
    fault_type: str = "",
) -> dict:
    healthy_ref = healthy["ref"]
    healthy_sched = healthy["sched"]
    faulty_flows, meta = _build_faulty_flows(pattern, rows, cols, fault, root=root)
    faulty_sched = schedule_collective_fair(faulty_flows)

    z_healthy = healthy_sched["makespan"] or healthy_ref["z_scheduled"]
    z_faulty = faulty_sched["makespan"]

    n = rows * cols
    full_flow_counts = {
        "broadcast": len(healthy["flows"]),
        "reduce": n - 1,
        "allreduce": len(healthy["flows"]),
        "allgather": len(healthy["flows"]),
        "gather": n - 1,
        "alltoall": n * (n - 1),
    }
    expected_full = full_flow_counts.get(pattern.lower(), len(healthy["flows"]))
    flow_count_faulty = len(faulty_flows)
    partial = flow_count_faulty < expected_full

    if z_healthy and z_faulty and not partial:
        ratio = round(z_faulty / z_healthy, 3)
    elif z_healthy and z_faulty and partial:
        ratio = round(max(1.0, z_faulty / z_healthy), 3)
    else:
        ratio = None

    note_parts: list[str] = []
    if not meta.get("reachable", True):
        note_parts.append("不可达/无流")
    if partial:
        note_parts.append(f"部分流({flow_count_faulty}/{expected_full})")
    if meta.get("hamilton_degraded"):
        note_parts.append("哈密顿环→路径")
    if not faulty_sched["ok"]:
        note_parts.append(f"调度冲突:{faulty_sched.get('collision')}")

    return {
        "pattern": pattern.lower(),
        "rows": rows,
        "cols": cols,
        "mesh": f"{rows}×{cols}",
        "root": root,
        "region": region,
        "fault_type": fault_type,
        "bad_nodes": sorted(fault.bad_nodes),
        "bad_edges": sorted(fault.bad_edges),
        "z_healthy_ref": healthy_ref["z_scheduled"],
        "z_healthy_scheduled": z_healthy,
        "z_faulty": z_faulty,
        "ratio": ratio,
        "stall_healthy": healthy_sched["stall"],
        "stall_faulty": faulty_sched["stall"],
        "peak_healthy": healthy_sched["peak"],
        "peak_faulty": faulty_sched["peak"],
        "ok_healthy": healthy_sched["ok"],
        "ok_faulty": faulty_sched["ok"],
        "reachable": meta.get("reachable", True),
        "hamilton_degraded": meta.get("hamilton_degraded", False),
        "is_cycle": meta.get("is_cycle", True),
        "flow_count_faulty": flow_count_faulty,
        "flow_count_expected": expected_full,
        "partial_coverage": partial,
        "note": "; ".join(note_parts) if note_parts else "",
    }

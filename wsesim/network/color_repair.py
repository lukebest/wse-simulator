"""Partial-good color route repair on defective mesh."""

from __future__ import annotations

from collections import deque

from wsesim.network.color import ColorPlan
from wsesim.network.color_routes import mesh_dims, xy_path
from wsesim.fault.defect_map import DefectMap


def prune_graph(
    graph: dict[int, list[int]],
    defect: DefectMap,
) -> dict[int, list[int]]:
    """Return graph with dead nodes/links removed."""
    dead_n = defect.dead_cores
    dead_l = defect.dead_links
    pruned: dict[int, list[int]] = {}
    for src, dsts in graph.items():
        if src in dead_n:
            continue
        pruned[src] = [d for d in dsts if d not in dead_n and (src, d) not in dead_l]
    return pruned


def _shortest_path(src: int, dst: int, graph: dict[int, list[int]]) -> list[int] | None:
    if src == dst:
        return [src]
    queue = deque([src])
    parent = {src: -1}
    while queue:
        node = queue.popleft()
        for nb in graph.get(node, []):
            if nb not in parent:
                parent[nb] = node
                if nb == dst:
                    queue.clear()
                    break
                queue.append(nb)
    if dst not in parent:
        return None
    path = []
    hop = dst
    while hop != -1:
        path.append(hop)
        hop = parent[hop]
    path.reverse()
    return path


def repair_color_plan(
    plan: ColorPlan,
    graph: dict[int, list[int]],
    defect: DefectMap,
) -> tuple[ColorPlan, float]:
    """Recompute per-color routes on pruned graph; return (plan, coverage fraction)."""
    pruned = prune_graph(graph, defect)
    rows, cols = mesh_dims(len(graph), None)
    repaired = ColorPlan.empty(len(graph), plan.num_colors)
    total_routes = 0
    successful = 0

    # Collect (src,dst) pairs from original plan per color
    for color_id in range(plan.num_colors):
        pairs: set[tuple[int, int]] = set()
        for node, colors in plan.dest.items():
            hops = colors.get(color_id, frozenset())
            for nxt in hops:
                pairs.add((node, nxt))
        # Rebuild routes: for each edge in old plan, try shortest path in pruned graph
        for src, dst in pairs:
            total_routes += 1
            if src not in pruned or dst not in pruned:
                continue
            path = _shortest_path(src, dst, pruned)
            if path is None:
                # Try detour via row/column redundancy: alternate XY/YX
                path = xy_path(src, dst, rows, cols)
                valid = all(
                    (path[i], path[i + 1]) in {(s, d) for s, ds in pruned.items() for d in ds}
                    or (path[i], path[i + 1]) not in defect.dead_links
                    for i in range(len(path) - 1)
                    if path[i] in pruned
                )
                if not valid:
                    continue
            else:
                for i in range(len(path) - 1):
                    if path[i] in pruned and path[i + 1] in pruned.get(path[i], []):
                        repaired.set_next_hops(path[i], color_id, frozenset({path[i + 1]}))
                successful += 1
                continue
            for i in range(len(path) - 1):
                if path[i] in pruned:
                    nxt = path[i + 1]
                    if nxt in pruned.get(path[i], []):
                        repaired.set_next_hops(path[i], color_id, frozenset({nxt}))
            successful += 1

    coverage = successful / max(1, total_routes)
    return repaired, coverage


def routable_nodes(graph: dict[int, list[int]], defect: DefectMap) -> set[int]:
    """Nodes reachable in largest connected component after defects."""
    pruned = prune_graph(graph, defect)
    if not pruned:
        return set()
    visited: set[int] = set()
    best: set[int] = set()

    for start in pruned:
        if start in visited:
            continue
        component: set[int] = set()
        queue = deque([start])
        while queue:
            n = queue.popleft()
            if n in component:
                continue
            component.add(n)
            for nb in pruned.get(n, []):
                if nb not in component:
                    queue.append(nb)
        visited |= component
        if len(component) > len(best):
            best = component
    return best

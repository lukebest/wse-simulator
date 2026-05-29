"""Partial-good color route repair on pruned mesh."""

from __future__ import annotations

from collections import deque

from wsesim.fault.defect_map import DefectMap
from wsesim.network.color import ColorPlan
from wsesim.network.color_routes import add_path_color, empty_plan
from wsesim.network.routing.table_based import TableBasedRouting


def repair_color_plan(
    plan: ColorPlan,
    graph: dict[int, list[int]],
    dead_nodes: set[int],
    dead_links: set[tuple[int, int]],
) -> tuple[ColorPlan, float]:
    """Recompute unicast routes on pruned graph; return repaired plan and coverage fraction."""
    alive = set(graph.keys()) - dead_nodes
    if not alive:
        return plan, 0.0

    pruned: dict[int, list[int]] = {}
    for src in alive:
        pruned[src] = [
            d for d in graph.get(src, []) if d in alive and (src, d) not in dead_links
        ]

    routing = TableBasedRouting(pruned)
    repaired = empty_plan(plan.rows, plan.cols, plan.num_colors)
    repaired.colors = list(plan.colors)
    repaired.unicast_modes = dict(plan.unicast_modes)

    routable_pairs = 0
    total_pairs = len(alive) * (len(alive) - 1)

    for node in alive:
        repaired.dest[node] = {}
        for c in range(plan.num_colors):
            orig = plan.dest.get(node, {}).get(c, set())
            valid = {h for h in orig if h in pruned.get(node, [])}
            if valid:
                repaired.dest[node][c] = valid
            elif c in plan.unicast_modes:
                repaired.set_unicast_mode(c, plan.unicast_modes[c])
            else:
                repaired.dest[node][c] = set()

    for src in alive:
        for dst in alive:
            if src == dst:
                continue
            try:
                routing.next_hop(src, dst, pruned)
                routable_pairs += 1
            except ValueError:
                continue

    coverage = routable_pairs / max(1, total_pairs)
    return repaired, coverage


def apply_defect_map_to_graph(
    graph: dict[int, list[int]], defect: DefectMap
) -> dict[int, list[int]]:
    pruned = {n: list(neighbors) for n, neighbors in graph.items() if n not in defect.dead_cores}
    for src in list(pruned):
        pruned[src] = [
            d
            for d in pruned[src]
            if d not in defect.dead_cores and (src, d) not in defect.dead_links
        ]
    for dead in defect.dead_cores:
        pruned.pop(dead, None)
    return pruned


def largest_component_nodes(graph: dict[int, list[int]]) -> set[int]:
    """Return nodes in largest connected component (undirected view)."""
    if not graph:
        return set()
    undirected: dict[int, set[int]] = {n: set() for n in graph}
    for src, dsts in graph.items():
        for d in dsts:
            undirected.setdefault(src, set()).add(d)
            undirected.setdefault(d, set()).add(src)
    visited: set[int] = set()
    best: set[int] = set()
    for start in undirected:
        if start in visited:
            continue
        comp: set[int] = set()
        queue = deque([start])
        while queue:
            n = queue.popleft()
            if n in comp:
                continue
            comp.add(n)
            for nb in undirected.get(n, set()):
                if nb not in comp and nb in graph:
                    queue.append(nb)
        visited |= comp
        if len(comp) > len(best):
            best = comp
    return best

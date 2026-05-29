"""Color allocation / optimization for flow-to-VN mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass(slots=True)
class FlowSpec:
    name: str
    start_cycle: int
    end_cycle: int
    links_used: set[tuple[int, int]] = field(default_factory=set)


@dataclass(slots=True)
class AllocationResult:
    flow_to_color: dict[str, int]
    peak_link_load: int
    num_colors: int
    strategy: str


def flows_conflict(a: FlowSpec, b: FlowSpec) -> bool:
    if a.end_cycle < b.start_cycle or b.end_cycle < a.start_cycle:
        return False
    return bool(a.links_used & b.links_used)


def greedy_allocate(flows: list[FlowSpec], num_colors: int) -> AllocationResult:
    """Assign flows to colors minimizing conflicts; reuse colors when non-overlapping."""
    flow_to_color: dict[str, int] = {}
    color_flows: dict[int, list[FlowSpec]] = defaultdict(list)
    link_load: dict[tuple[int, int], int] = defaultdict(int)

    for flow in sorted(flows, key=lambda f: (f.start_cycle, -len(f.links_used))):
        best_color = None
        best_load = 10**9
        for c in range(num_colors):
            if any(flows_conflict(flow, f) for f in color_flows[c]):
                continue
            peak = max(
                (link_load[l] for l in flow.links_used),
                default=0,
            )
            if peak < best_load:
                best_load = peak
                best_color = c
        if best_color is None:
            best_color = min(
                range(num_colors),
                key=lambda c: sum(1 for f in color_flows[c] if flows_conflict(flow, f)),
            )
        flow_to_color[flow.name] = best_color
        color_flows[best_color].append(flow)
        for link in flow.links_used:
            link_load[link] += 1

    peak = max(link_load.values()) if link_load else 0
    return AllocationResult(
        flow_to_color=flow_to_color,
        peak_link_load=peak,
        num_colors=num_colors,
        strategy="greedy",
    )


def graph_coloring_allocate(flows: list[FlowSpec], num_colors: int) -> AllocationResult:
    """Conflict-graph coloring with temporal overlap as edges."""
    names = [f.name for f in flows]
    conflicts: dict[str, set[str]] = {n: set() for n in names}
    by_name = {f.name: f for f in flows}
    for i, a in enumerate(flows):
        for b in flows[i + 1 :]:
            if flows_conflict(a, b):
                conflicts[a.name].add(b.name)
                conflicts[b.name].add(a.name)

    order = sorted(names, key=lambda n: -len(conflicts[n]))
    flow_to_color: dict[str, int] = {}
    for name in order:
        used = {flow_to_color[n] for n in conflicts[name] if n in flow_to_color}
        c = next((c for c in range(num_colors) if c not in used), 0)
        flow_to_color[name] = c

    link_load: dict[tuple[int, int], int] = defaultdict(int)
    for f in flows:
        for link in f.links_used:
            link_load[link] += 1
    return AllocationResult(
        flow_to_color=flow_to_color,
        peak_link_load=max(link_load.values()) if link_load else 0,
        num_colors=num_colors,
        strategy="graph_coloring",
    )


def ilp_allocate(flows: list[FlowSpec], num_colors: int) -> AllocationResult:
    """ILP minimize peak link load; fall back to greedy if pulp unavailable."""
    try:
        import pulp
    except ImportError:
        return greedy_allocate(flows, num_colors)

    prob = pulp.LpProblem("color_alloc", pulp.LpMinimize)
    x = {
        (f.name, c): pulp.LpVariable(f"x_{f.name}_{c}", cat="Binary")
        for f in flows
        for c in range(num_colors)
    }
    load = {
        link: pulp.LpVariable(f"load_{link[0]}_{link[1]}", lowBound=0, cat="Integer")
        for link in {l for f in flows for l in f.links_used}
    }
    peak = pulp.LpVariable("peak_load", lowBound=0, cat="Integer")
    prob += peak

    for f in flows:
        prob += pulp.lpSum(x[f.name, c] for c in range(num_colors)) == 1

    for i, a in enumerate(flows):
        for b in flows[i + 1 :]:
            if flows_conflict(a, b):
                for c in range(num_colors):
                    prob += x[a.name, c] + x[b.name, c] <= 1

    all_links = {l for f in flows for l in f.links_used}
    for link in all_links:
        prob += load[link] == pulp.lpSum(
            x[f.name, c]
            for f in flows
            for c in range(num_colors)
            if link in f.links_used
        )
        prob += load[link] <= peak

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    flow_to_color: dict[str, int] = {}
    for f in flows:
        for c in range(num_colors):
            if pulp.value(x[f.name, c]) and pulp.value(x[f.name, c]) > 0.5:
                flow_to_color[f.name] = c
                break
        else:
            flow_to_color[f.name] = 0

    return AllocationResult(
        flow_to_color=flow_to_color,
        peak_link_load=int(pulp.value(peak) or 0),
        num_colors=num_colors,
        strategy="ilp",
    )


def is_acyclic_route(dest: dict[int, dict[int, set[int]]], color: int, num_nodes: int) -> bool:
    """Detect cycle in static forwarding graph for one color."""
    graph: dict[int, list[int]] = {n: list(dest.get(n, {}).get(color, set())) for n in range(num_nodes)}
    visited = set()
    stack = set()

    def dfs(node: int) -> bool:
        if node in stack:
            return False
        if node in visited:
            return True
        visited.add(node)
        stack.add(node)
        for nxt in graph.get(node, []):
            if not dfs(nxt):
                return False
        stack.remove(node)
        return True

    for n in range(num_nodes):
        if n not in visited:
            if not dfs(n):
                return False
    return True

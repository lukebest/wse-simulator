"""Color allocation and optimization for virtual networks."""

from __future__ import annotations

from dataclasses import dataclass, field

from wsesim.network.color import ColorPlan
from wsesim.network.color_routes import (
    build_default_mixed_plan,
    build_path_color,
    is_acyclic_color,
    mesh_dims,
)


@dataclass(slots=True)
class FlowSpec:
    """A communication flow requiring a color assignment."""

    flow_id: str
    src: int
    dst: int
    pattern: str
    phase: int = 0
    concurrent_with: frozenset[str] = frozenset()


@dataclass(slots=True)
class ColorAllocation:
    """Maps flows to colors and holds the merged ColorPlan."""

    flow_to_color: dict[str, int] = field(default_factory=dict)
    plan: ColorPlan | None = None
    peak_link_load: float = 0.0
    deadlock_free: bool = True


def _flows_conflict(a: FlowSpec, b: FlowSpec) -> bool:
    if a.flow_id in b.concurrent_with or b.flow_id in a.concurrent_with:
        return True
    if a.phase == b.phase:
        return True
    return False


def allocate_greedy(
    flows: list[FlowSpec],
    num_nodes: int,
    num_colors: int = 16,
    cols: int | None = None,
) -> ColorAllocation:
    """Greedy color assignment: conflicting flows get distinct colors."""
    rows, cols = mesh_dims(num_nodes, cols)
    plan = ColorPlan.empty(num_nodes, num_colors)
    flow_to_color: dict[str, int] = {}
    color_usage: dict[int, list[str]] = {c: [] for c in range(num_colors)}

    for flow in flows:
        assigned = None
        for c in range(num_colors):
            conflict = any(
                _flows_conflict(flow, other_flow)
                for other_id in color_usage[c]
                for other_flow in flows
                if other_flow.flow_id == other_id
            )
            if not conflict:
                build_path_color(plan, c, flow.src, flow.dst, rows, cols)
                if is_acyclic_color(plan, c) or flow.pattern == "ring":
                    assigned = c
                    color_usage[c].append(flow.flow_id)
                    break
        if assigned is None:
            assigned = len(flow_to_color) % num_colors
            build_path_color(plan, assigned, flow.src, flow.dst, rows, cols)
            color_usage[assigned].append(flow.flow_id)
        flow_to_color[flow.flow_id] = assigned

    return ColorAllocation(flow_to_color=flow_to_color, plan=plan, deadlock_free=True)


def allocate_graph_coloring(
    flows: list[FlowSpec],
    num_nodes: int,
    num_colors: int = 16,
    cols: int | None = None,
) -> ColorAllocation:
    """Graph-color flows by conflict; build path per (flow, color)."""
    rows, cols = mesh_dims(num_nodes, cols)
    plan = ColorPlan.empty(num_nodes, num_colors)
    # Build conflict graph
    colors_for_flow: dict[str, int] = {}
    used: dict[int, set[int]] = {c: set() for c in range(num_colors)}

    for flow in flows:
        forbidden: set[int] = set()
        for fid, c in colors_for_flow.items():
            other = next(f for f in flows if f.flow_id == fid)
            if _flows_conflict(flow, other):
                forbidden.add(c)
        chosen = next((c for c in range(num_colors) if c not in forbidden), 0)
        colors_for_flow[flow.flow_id] = chosen
        build_path_color(plan, chosen, flow.src, flow.dst, rows, cols)
        used[chosen].add(flow.src)

    return ColorAllocation(flow_to_color=colors_for_flow, plan=plan)


def allocate_ilp(
    flows: list[FlowSpec],
    num_nodes: int,
    num_colors: int = 16,
    cols: int | None = None,
) -> ColorAllocation:
    """ILP allocation via pulp; falls back to graph coloring if pulp unavailable."""
    try:
        import pulp
    except ImportError:
        return allocate_graph_coloring(flows, num_nodes, num_colors, cols)

    rows, cols = mesh_dims(num_nodes, cols)
    n_flows = len(flows)
    if n_flows == 0:
        return ColorAllocation(plan=build_default_mixed_plan(num_nodes, num_colors, cols))

    prob = pulp.LpProblem("color_alloc", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", ((i, c) for i in range(n_flows) for c in range(num_colors)), cat="Binary")
    # Each flow exactly one color
    for i in range(n_flows):
        prob += pulp.lpSum(x[i, c] for c in range(num_colors)) == 1
    # Conflicting flows cannot share color
    for i in range(n_flows):
        for j in range(i + 1, n_flows):
            if _flows_conflict(flows[i], flows[j]):
                for c in range(num_colors):
                    prob += x[i, c] + x[j, c] <= 1
    # Minimize max color usage (balance)
    load = pulp.LpVariable("peak_load", lowBound=0)
    for c in range(num_colors):
        prob += pulp.lpSum(x[i, c] for i in range(n_flows)) <= load
    prob += load

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    plan = ColorPlan.empty(num_nodes, num_colors)
    flow_to_color: dict[str, int] = {}
    for i, flow in enumerate(flows):
        chosen = 0
        for c in range(num_colors):
            if pulp.value(x[i, c]) and pulp.value(x[i, c]) > 0.5:
                chosen = c
                break
        flow_to_color[flow.flow_id] = chosen
        build_path_color(plan, chosen, flow.src, flow.dst, rows, cols)

    return ColorAllocation(flow_to_color=flow_to_color, plan=plan)


def build_mixed_workload_plan(num_nodes: int, num_colors: int = 16, cols: int | None = None) -> ColorPlan:
    """Default optimized mixed-workload color plan."""
    return build_default_mixed_plan(num_nodes, num_colors, cols)

"""Pluggable color planners for TDM flattened-butterfly overlays (DSE)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import pstdev
from typing import Protocol

from wsesim.network.collective import _resolve_groups_by_dimension
from wsesim.network.tdm_coloring import (
    ColorPlan,
    LogicalLink,
    PhysicalLink,
    _build_link_active,
    assign_colors,
    assign_colors_for_routes,
    route_color,
)

CONSTRAINT_A = "A"
CONSTRAINT_AB = "A+B"
LINK_UNIVERSE_FULL = "full"
LINK_UNIVERSE_ND_ACTIVE = "nd_active"

PLANNER_GREEDY_FIRST_FIT = "greedy_first_fit"
PLANNER_GREEDY_BEST_FIT = "greedy_best_fit_balanced"
PLANNER_EDGE_ROUND_ROBIN = "per_edge_round_robin"
PLANNER_ILP_MIN_C = "ilp_min_C"
PLANNER_ILP_MIN_C_BALANCE = "ilp_min_C_then_balance"

ALL_PLANNERS = (
    PLANNER_GREEDY_FIRST_FIT,
    PLANNER_GREEDY_BEST_FIT,
    PLANNER_EDGE_ROUND_ROBIN,
    PLANNER_ILP_MIN_C,
    PLANNER_ILP_MIN_C_BALANCE,
)


class ColorableOverlay(Protocol):
    n: int

    def logical_links(self) -> list[tuple[int, int, int]]: ...
    def physical_path(self, src: int, dst: int) -> list[PhysicalLink]: ...
    def physical_links(self) -> list[PhysicalLink]: ...
    def dim_order_route(self, src: int, dst: int) -> list[LogicalLink]: ...


@dataclass(slots=True)
class ColorPlannerConfig:
    constraint: str = CONSTRAINT_A
    link_universe: str = LINK_UNIVERSE_FULL
    planner: str = PLANNER_GREEDY_FIRST_FIT
    time_limit_s: float = 60.0
    topology_hint: dict | None = None
    ilp_fallback: str = PLANNER_GREEDY_BEST_FIT


@dataclass(slots=True)
class NdAllGatherRoutes:
    pairs: list[tuple[int, int]]
    route_paths: dict[tuple[int, int], list[LogicalLink]]
    active_logical_links: set[LogicalLink]
    monochrome_paths: list[list[LogicalLink]]


@dataclass(slots=True)
class PlanValidation:
    edge_conflicts: list[PhysicalLink]
    missing_logical: set[LogicalLink]
    extra_logical: set[LogicalLink]
    monochrome_fail_pairs: list[tuple[int, int]]
    monochrome_rate: float


@dataclass(slots=True)
class PlanStats:
    C: int
    color_lower_bound: int
    min_links_per_color: int
    max_links_per_color: int
    mean_links_per_color: float
    std_links_per_color: float
    balance_ratio: float
    planner_used: str
    ilp_fallback_used: bool = False


def topology_hint_for_overlay(topo: ColorableOverlay) -> dict:
    k = getattr(topo, "k", None)
    n = topo.n
    if k is not None and k > 1:
        return {"k": int(k), "n": int(n)}
    k_dims = getattr(topo, "k_dims", None)
    rows = getattr(topo, "rows", None)
    cols = getattr(topo, "cols", None)
    if k_dims is not None:
        hint: dict = {"k_dims": list(k_dims)}
        if rows is not None and cols is not None:
            hint["rows"] = int(rows)
            hint["cols"] = int(cols)
        return hint
    if rows is not None and cols is not None:
        return {"rows": int(rows), "cols": int(cols), "n": n}
    return {"n": n}


def node_ids_for_overlay(topo: ColorableOverlay) -> list[int]:
    node_ids = getattr(topo, "node_ids", None)
    if callable(node_ids):
        return list(node_ids())
    rows = getattr(topo, "rows", None)
    cols = getattr(topo, "cols", None)
    if rows is not None and cols is not None:
        return list(range(int(rows) * int(cols)))
    parent_cols = getattr(topo, "parent_cols", None)
    keep_rows = getattr(topo, "keep_rows", None)
    keep_cols = getattr(topo, "keep_cols", None)
    if parent_cols and keep_rows and keep_cols:
        return [
            r * int(parent_cols) + c
            for r in range(int(keep_rows))
            for c in range(int(keep_cols))
        ]
    raise ValueError("Cannot infer node ids for overlay topology.")


def enumerate_nd_allgather_routes(
    topo: ColorableOverlay,
    nodes: list[int] | None = None,
    topology_hint: dict | None = None,
) -> NdAllGatherRoutes:
    """Enumerate ND dimension-exchange AllGather (src, dst) routes on *topo*."""
    if nodes is None:
        nodes = node_ids_for_overlay(topo)
    hint = topology_hint if topology_hint is not None else topology_hint_for_overlay(topo)
    groups_by_dim = _resolve_groups_by_dimension(nodes, hint)

    pairs: list[tuple[int, int]] = []
    route_paths: dict[tuple[int, int], list[LogicalLink]] = {}
    active: set[LogicalLink] = set()
    seen_paths: set[tuple[LogicalLink, ...]] = set()
    monochrome_paths: list[list[LogicalLink]] = []

    for groups in groups_by_dim.values():
        for group in groups:
            for src in group:
                for dst in group:
                    if src == dst:
                        continue
                    pair = (src, dst)
                    if pair not in route_paths:
                        path = topo.dim_order_route(src, dst)
                        route_paths[pair] = path
                        key = tuple(path)
                        if path and key not in seen_paths:
                            seen_paths.add(key)
                            monochrome_paths.append(path)
                    pairs.append(pair)
                    active.update(route_paths[pair])

    return NdAllGatherRoutes(
        pairs=pairs,
        route_paths=route_paths,
        active_logical_links=active,
        monochrome_paths=monochrome_paths,
    )


def plan_stats(plan: ColorPlan, planner_used: str = "", ilp_fallback_used: bool = False) -> PlanStats:
    counts = Counter(plan.color_of_logical.values())
    if not counts:
        vals = [0]
    else:
        vals = list(counts.values())
    mn, mx = min(vals), max(vals)
    mean = sum(vals) / len(vals) if vals else 0.0
    std = pstdev(vals) if len(vals) > 1 else 0.0
    return PlanStats(
        C=plan.C,
        color_lower_bound=plan.color_lower_bound,
        min_links_per_color=mn,
        max_links_per_color=mx,
        mean_links_per_color=mean,
        std_links_per_color=std,
        balance_ratio=(mx / mn) if mn > 0 else 1.0,
        planner_used=planner_used,
        ilp_fallback_used=ilp_fallback_used,
    )


def validate_plan(
    plan: ColorPlan,
    topo: ColorableOverlay,
    routes: NdAllGatherRoutes | None = None,
) -> PlanValidation:
    """Validate Constraint A and optional route-monochrome feasibility."""
    edge_conflicts: list[PhysicalLink] = []
    logical_expected = {(u, v) for u, v, _ in topo.logical_links()}
    logical_seen = set(plan.color_of_logical.keys())

    for logical, color in plan.color_of_logical.items():
        for edge in topo.physical_path(*logical):
            owner = plan.link_active.get(edge, [None] * plan.C)[color]
            if owner != logical:
                edge_conflicts.append(edge)

    for color in range(plan.C):
        used: set[PhysicalLink] = set()
        for edge, owners in plan.link_active.items():
            if owners[color] is None:
                continue
            if edge in used:
                edge_conflicts.append(edge)
            used.add(edge)

    mono_fail: list[tuple[int, int]] = []
    if routes is not None:
        for src, dst in routes.pairs:
            path = routes.route_paths.get((src, dst), topo.dim_order_route(src, dst))
            if path and route_color(plan, path) is None:
                mono_fail.append((src, dst))

    total_pairs = len(routes.pairs) if routes else 0
    mono_rate = (
        1.0
        if total_pairs == 0
        else (total_pairs - len(mono_fail)) / total_pairs
    )

    return PlanValidation(
        edge_conflicts=edge_conflicts,
        missing_logical=logical_expected - logical_seen,
        extra_logical=logical_seen - logical_expected,
        monochrome_fail_pairs=mono_fail,
        monochrome_rate=mono_rate,
    )


def _prepare_link_sets(
    topo: ColorableOverlay,
    config: ColorPlannerConfig,
) -> tuple[
    list[tuple[int, int, int]],
    dict[LogicalLink, list[PhysicalLink]],
    list[PhysicalLink],
    set[LogicalLink],
    NdAllGatherRoutes,
    list[list[LogicalLink]],
]:
    all_links = topo.logical_links()
    paths = {(u, v): topo.physical_path(u, v) for u, v, _ in all_links}
    physical = topo.physical_links()
    routes = enumerate_nd_allgather_routes(
        topo,
        topology_hint=config.topology_hint,
    )
    target = {link for link, _, _ in all_links}
    if config.link_universe == LINK_UNIVERSE_ND_ACTIVE:
        target = routes.active_logical_links
    mono_paths = routes.monochrome_paths if config.constraint == CONSTRAINT_AB else []
    return all_links, paths, physical, target, routes, mono_paths


def _load_per_physical(
    logical_links: list[tuple[int, int, int]],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    physical_links: list[PhysicalLink],
) -> dict[PhysicalLink, int]:
    load: dict[PhysicalLink, int] = {edge: 0 for edge in physical_links}
    for u, v, _ in logical_links:
        for edge in physical_paths.get((u, v), []):
            load[edge] = load.get(edge, 0) + 1
    return load


def _finalize_plan(
    color_of_logical: dict[LogicalLink, int],
    logical_dims: dict[LogicalLink, int],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    physical_links: list[PhysicalLink],
    load_per_link: dict[PhysicalLink, int],
    lower_bound: int,
) -> ColorPlan:
    color_count = max(color_of_logical.values()) + 1 if color_of_logical else 1
    link_active = _build_link_active(color_of_logical, physical_paths, color_count)
    return ColorPlan(
        C=color_count,
        link_active=link_active,
        color_of_logical=color_of_logical,
        logical_dims=logical_dims,
        load_per_physical_link=load_per_link,
        color_lower_bound=lower_bound,
    )


def _seed_free_bitmap(
    color_of_logical: dict[LogicalLink, int],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    color_count: int,
) -> dict[PhysicalLink, list[bool]]:
    free_at: dict[PhysicalLink, list[bool]] = defaultdict(lambda: [True] * color_count)
    for link, color in color_of_logical.items():
        for edge in physical_paths.get(link, []):
            free_at[edge][color] = False
    return dict(free_at)


def assign_colors_best_fit(
    logical_links: list[tuple[int, int, int]],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    physical_links: list[PhysicalLink],
    *,
    target_links: set[LogicalLink] | None = None,
    force_monochrome_paths: list[list[LogicalLink]] | None = None,
) -> ColorPlan:
    """Greedy coloring: pick feasible color with smallest current load."""
    if force_monochrome_paths:
        return assign_colors_for_routes(
            logical_links,
            physical_paths,
            physical_links,
            force_monochrome_paths,
        )

    if not logical_links:
        return assign_colors(
            logical_links,
            physical_paths,
            physical_links,
        )

    if target_links is None:
        target_links = {(u, v) for u, v, _ in logical_links}

    logical_with_dim = [((u, v), dim) for u, v, dim in logical_links]
    logical_dims_map = {link: dim for link, dim in logical_with_dim}
    load_per_link = _load_per_physical(logical_links, physical_paths, physical_links)
    lower_bound = max([1, *load_per_link.values()])

    color_count = lower_bound
    color_of_logical: dict[LogicalLink, int] = {}
    free_at: dict[PhysicalLink, list[bool]] = {
        edge: [True] * color_count for edge in load_per_link
    }
    color_load: Counter[int] = Counter()

    def place_link(logical_link: LogicalLink, dim: int) -> bool:
        nonlocal color_count
        path = physical_paths.get(logical_link, [])
        feasible = []
        for color in range(color_count):
            if all(free_at.setdefault(edge, [True] * color_count)[color] for edge in path):
                feasible.append(color)
        if not feasible:
            for edge in free_at:
                free_at[edge].append(True)
            color_count += 1
            feasible = [color_count - 1]
        chosen = min(feasible, key=lambda c: (color_load[c], c))
        color_of_logical[logical_link] = chosen
        logical_dims_map[logical_link] = dim
        color_load[chosen] += 1
        for edge in path:
            free_at.setdefault(edge, [True] * color_count)
            free_at[edge][chosen] = False
        return True

    target_items = [(link, dim) for link, dim in logical_with_dim if link in target_links]
    rest_items = [(link, dim) for link, dim in logical_with_dim if link not in target_links]

    ordered_target = sorted(
        target_items,
        key=lambda item: -len(physical_paths.get(item[0], [])),
    )
    ordered_rest = sorted(
        rest_items,
        key=lambda item: -len(physical_paths.get(item[0], [])),
    )
    for logical_link, dim in ordered_target + ordered_rest:
        place_link(logical_link, dim)

    return _finalize_plan(
        color_of_logical,
        logical_dims_map,
        physical_paths,
        physical_links,
        load_per_link,
        lower_bound,
    )


def assign_colors_edge_round_robin(
    logical_links: list[tuple[int, int, int]],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    physical_links: list[PhysicalLink],
    *,
    target_links: set[LogicalLink] | None = None,
    force_monochrome_paths: list[list[LogicalLink]] | None = None,
) -> ColorPlan:
    """Per-edge round-robin slot hints merged via best-fit placement."""
    if force_monochrome_paths:
        return assign_colors_for_routes(
            logical_links,
            physical_paths,
            physical_links,
            force_monochrome_paths,
        )

    if not logical_links:
        return assign_colors(logical_links, physical_paths, physical_links)

    if target_links is None:
        target_links = {(u, v) for u, v, _ in logical_links}

    logical_with_dim = [((u, v), dim) for u, v, dim in logical_links]
    logical_dims_map = {link: dim for link, dim in logical_with_dim}
    load_per_link = _load_per_physical(logical_links, physical_paths, physical_links)
    lower_bound = max([1, *load_per_link.values()])

    edge_to_links: dict[PhysicalLink, list[LogicalLink]] = defaultdict(list)
    for link in target_links:
        for edge in physical_paths.get(link, []):
            edge_to_links[edge].append(link)

    slot_hints: dict[LogicalLink, int] = {}
    for edge in sorted(edge_to_links, key=lambda e: -load_per_link.get(e, 0)):
        for slot, link in enumerate(sorted(edge_to_links[edge])):
            slot_hints[link] = max(slot_hints.get(link, 0), slot)

    color_count = lower_bound
    color_of_logical: dict[LogicalLink, int] = {}
    free_at: dict[PhysicalLink, list[bool]] = {
        edge: [True] * color_count for edge in load_per_link
    }
    color_load: Counter[int] = Counter()

    def place_link(logical_link: LogicalLink, dim: int) -> None:
        nonlocal color_count
        path = physical_paths.get(logical_link, [])
        hint = slot_hints.get(logical_link, 0)
        feasible = []
        for color in range(color_count):
            if all(free_at.setdefault(edge, [True] * color_count)[color] for edge in path):
                feasible.append(color)
        if not feasible:
            for edge in free_at:
                free_at[edge].append(True)
            color_count += 1
            feasible = [color_count - 1]
        chosen = min(
            feasible,
            key=lambda c: (color_load[c], abs(c - (hint % max(1, color_count))), c),
        )
        color_of_logical[logical_link] = chosen
        logical_dims_map[logical_link] = dim
        color_load[chosen] += 1
        for edge in path:
            free_at.setdefault(edge, [True] * color_count)
            free_at[edge][chosen] = False

    target_items = [(link, dim) for link, dim in logical_with_dim if link in target_links]
    rest_items = [(link, dim) for link, dim in logical_with_dim if link not in target_links]
    for logical_link, dim in sorted(
        target_items + rest_items,
        key=lambda item: -len(physical_paths.get(item[0], [])),
    ):
        place_link(logical_link, dim)

    return _finalize_plan(
        color_of_logical,
        logical_dims_map,
        physical_paths,
        physical_links,
        load_per_link,
        lower_bound,
    )


def _assign_with_ilp(
    logical_links: list[tuple[int, int, int]],
    physical_paths: dict[LogicalLink, list[PhysicalLink]],
    physical_links: list[PhysicalLink],
    *,
    target_links: set[LogicalLink],
    force_monochrome_paths: list[list[LogicalLink]],
    minimize_balance: bool,
    time_limit_s: float,
) -> ColorPlan | None:
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None

    logical_with_dim = [((u, v), dim) for u, v, dim in logical_links]
    logical_dims_map = {link: dim for link, dim in logical_with_dim}
    load_per_link = _load_per_physical(logical_links, physical_paths, physical_links)
    lower_bound = max([1, *load_per_link.values()])

    optimize_links = [link for link, _ in logical_with_dim if link in target_links]
    if not optimize_links:
        optimize_links = [link for link, _ in logical_with_dim]

    edge_to_links: dict[PhysicalLink, list[LogicalLink]] = defaultdict(list)
    for link in optimize_links:
        for edge in physical_paths.get(link, []):
            edge_to_links[edge].append(link)

    max_c = lower_bound + len(optimize_links)
    chosen_c: int | None = None
    assignment: dict[LogicalLink, int] = {}

    for trial_c in range(lower_bound, max_c + 1):
        model = cp_model.CpModel()
        x: dict[LogicalLink, list[cp_model.IntVar]] = {}
        for link in optimize_links:
            vars_c = [model.NewBoolVar(f"x_{link[0]}_{link[1]}_{c}") for c in range(trial_c)]
            model.Add(sum(vars_c) == 1)
            x[link] = vars_c

        for edge, links_on_edge in edge_to_links.items():
            for c in range(trial_c):
                model.Add(sum(x[link][c] for link in links_on_edge if link in x) <= 1)

        for path in force_monochrome_paths:
            path_links = [hop for hop in path if hop in x]
            if len(path_links) < 2:
                continue
            first = path_links[0]
            for hop in path_links[1:]:
                for c in range(trial_c):
                    model.Add(x[first][c] == x[hop][c])

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        chosen_c = trial_c
        assignment = {
            link: next(c for c in range(trial_c) if solver.Value(x[link][c]))
            for link in optimize_links
        }
        break

    if chosen_c is None:
        return None

    if minimize_balance and chosen_c is not None:
        model = cp_model.CpModel()
        x = {}
        for link in optimize_links:
            vars_c = [model.NewBoolVar(f"x2_{link[0]}_{link[1]}_{c}") for c in range(chosen_c)]
            model.Add(sum(vars_c) == 1)
            x[link] = vars_c
        for edge, links_on_edge in edge_to_links.items():
            for c in range(chosen_c):
                model.Add(sum(x[link][c] for link in links_on_edge if link in x) <= 1)
        for path in force_monochrome_paths:
            path_links = [hop for hop in path if hop in x]
            if len(path_links) < 2:
                continue
            first = path_links[0]
            for hop in path_links[1:]:
                for c in range(chosen_c):
                    model.Add(x[first][c] == x[hop][c])

        loads = []
        for c in range(chosen_c):
            load_var = model.NewIntVar(0, len(optimize_links), f"load_{c}")
            model.Add(load_var == sum(x[link][c] for link in optimize_links))
            loads.append(load_var)
        max_load = model.NewIntVar(0, len(optimize_links), "max_load")
        for load_var in loads:
            model.Add(load_var <= max_load)
        model.Minimize(max_load)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        if solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assignment = {
                link: next(c for c in range(chosen_c) if solver.Value(x[link][c]))
                for link in optimize_links
            }

    color_of_logical = dict(assignment)
    assert chosen_c is not None
    free_at = _seed_free_bitmap(color_of_logical, physical_paths, chosen_c)
    color_load: Counter[int] = Counter(color_of_logical.values())

    rest = [(link, dim) for link, dim in logical_with_dim if link not in color_of_logical]
    for logical_link, dim in sorted(
        rest,
        key=lambda item: -len(physical_paths.get(item[0], [])),
    ):
        path = physical_paths.get(logical_link, [])
        feasible = []
        for color in range(chosen_c):
            if all(free_at.setdefault(edge, [True] * chosen_c)[color] for edge in path):
                feasible.append(color)
        if not feasible:
            for edge in free_at:
                free_at[edge].append(True)
            chosen_c += 1
            feasible = [chosen_c - 1]
        chosen = min(feasible, key=lambda c: (color_load[c], c))
        color_of_logical[logical_link] = chosen
        logical_dims_map[logical_link] = dim
        color_load[chosen] += 1
        for edge in path:
            free_at.setdefault(edge, [True] * chosen_c)
            free_at[edge][chosen] = False

    return _finalize_plan(
        color_of_logical,
        logical_dims_map,
        physical_paths,
        physical_links,
        load_per_link,
        lower_bound,
    )


def build_color_plan(
    topo: ColorableOverlay,
    config: ColorPlannerConfig,
) -> tuple[ColorPlan, PlanStats]:
    """Build a ColorPlan for *topo* using *config*."""
    all_links, paths, physical, target, routes, mono_paths = _prepare_link_sets(topo, config)
    planner_used = config.planner
    ilp_fallback = False

    if config.planner == PLANNER_GREEDY_FIRST_FIT:
        if config.constraint == CONSTRAINT_AB:
            plan = assign_colors_for_routes(all_links, paths, physical, mono_paths)
        elif config.link_universe == LINK_UNIVERSE_ND_ACTIVE:
            plan = assign_colors_best_fit(all_links, paths, physical, target_links=target)
        else:
            plan = assign_colors(all_links, paths, physical)

    elif config.planner == PLANNER_GREEDY_BEST_FIT:
        plan = assign_colors_best_fit(
            all_links,
            paths,
            physical,
            target_links=target if config.link_universe == LINK_UNIVERSE_ND_ACTIVE else None,
            force_monochrome_paths=mono_paths if config.constraint == CONSTRAINT_AB else None,
        )

    elif config.planner == PLANNER_EDGE_ROUND_ROBIN:
        plan = assign_colors_edge_round_robin(
            all_links,
            paths,
            physical,
            target_links=target if config.link_universe == LINK_UNIVERSE_ND_ACTIVE else None,
            force_monochrome_paths=mono_paths if config.constraint == CONSTRAINT_AB else None,
        )

    elif config.planner in (PLANNER_ILP_MIN_C, PLANNER_ILP_MIN_C_BALANCE):
        plan = _assign_with_ilp(
            all_links,
            paths,
            physical,
            target_links=target,
            force_monochrome_paths=mono_paths,
            minimize_balance=config.planner == PLANNER_ILP_MIN_C_BALANCE,
            time_limit_s=config.time_limit_s,
        )
        if plan is None:
            ilp_fallback = True
            planner_used = config.ilp_fallback
            fallback_cfg = ColorPlannerConfig(
                constraint=config.constraint,
                link_universe=config.link_universe,
                planner=config.ilp_fallback,
                topology_hint=config.topology_hint,
            )
            plan, _ = build_color_plan(topo, fallback_cfg)
    else:
        raise ValueError(f"Unknown planner: {config.planner}")

    stats = plan_stats(plan, planner_used=planner_used, ilp_fallback_used=ilp_fallback)
    return plan, stats

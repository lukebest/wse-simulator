"""Tests for color planner DSE helpers and planners."""

from __future__ import annotations

import pytest

from wsesim.network.color_planners import (
    CONSTRAINT_A,
    CONSTRAINT_AB,
    LINK_UNIVERSE_FULL,
    LINK_UNIVERSE_ND_ACTIVE,
    ColorPlannerConfig,
    ALL_PLANNERS,
    PLANNER_GREEDY_FIRST_FIT,
    PLANNER_ILP_MIN_C,
    build_color_plan,
    enumerate_nd_allgather_routes,
    plan_stats,
    validate_plan,
)
from wsesim.network.tdm_coloring import assign_colors, route_color
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly

pytestmark_ortools = pytest.mark.ortools


def _topo_k2_n4() -> TDMFlatButterfly:
    return TDMFlatButterfly(k=2, n=4, rows=4, cols=4)


def test_enumerate_nd_allgather_routes_k2_n4() -> None:
    topo = _topo_k2_n4()
    routes = enumerate_nd_allgather_routes(topo, topology_hint={"k": 2, "n": 4})
    assert routes.pairs
    assert routes.active_logical_links
    assert all(len(path) >= 1 for path in routes.route_paths.values())


def test_greedy_baseline_matches_assign_colors() -> None:
    topo = _topo_k2_n4()
    links = topo.logical_links()
    paths = {(u, v): topo.physical_path(u, v) for u, v, _ in links}
    expected = assign_colors(links, paths, topo.physical_links())
    config = ColorPlannerConfig(
        planner=PLANNER_GREEDY_FIRST_FIT,
        constraint=CONSTRAINT_A,
        link_universe=LINK_UNIVERSE_FULL,
        topology_hint={"k": 2, "n": 4},
    )
    plan, _ = build_color_plan(topo, config)
    assert plan.C == expected.C
    assert plan.color_of_logical == expected.color_of_logical


@pytest.mark.parametrize("planner", ALL_PLANNERS)
def test_all_planners_conflict_free_k2_n4(planner: str) -> None:
    if planner.startswith("ilp"):
        pytest.importorskip("ortools")
    topo = _topo_k2_n4()
    config = ColorPlannerConfig(
        planner=planner,
        constraint=CONSTRAINT_A,
        link_universe=LINK_UNIVERSE_FULL,
        topology_hint={"k": 2, "n": 4},
        time_limit_s=30.0,
    )
    plan, stats = build_color_plan(topo, config)
    routes = enumerate_nd_allgather_routes(topo, topology_hint={"k": 2, "n": 4})
    validation = validate_plan(plan, topo, routes)
    assert validation.edge_conflicts == []
    assert validation.missing_logical == set()
    assert plan.C >= plan.color_lower_bound
    assert stats.balance_ratio >= 1.0


@pytest.mark.parametrize("planner", ALL_PLANNERS)
def test_ab_constraint_monochrome_k2_n4(planner: str) -> None:
    if planner.startswith("ilp"):
        pytest.importorskip("ortools")
    topo = _topo_k2_n4()
    config = ColorPlannerConfig(
        planner=planner,
        constraint=CONSTRAINT_AB,
        link_universe=LINK_UNIVERSE_FULL,
        topology_hint={"k": 2, "n": 4},
        time_limit_s=30.0,
    )
    plan, _ = build_color_plan(topo, config)
    routes = enumerate_nd_allgather_routes(topo, topology_hint={"k": 2, "n": 4})
    validation = validate_plan(plan, topo, routes)
    assert validation.monochrome_rate == 1.0
    for path in routes.monochrome_paths:
        assert route_color(plan, path) is not None


def test_nd_active_subset_covers_routes() -> None:
    topo = TDMFlatButterfly(k=8, n=2, rows=8, cols=8)
    routes = enumerate_nd_allgather_routes(topo, topology_hint={"k": 8, "n": 2})
    config = ColorPlannerConfig(
        planner=PLANNER_GREEDY_FIRST_FIT,
        constraint=CONSTRAINT_A,
        link_universe=LINK_UNIVERSE_ND_ACTIVE,
        topology_hint={"k": 8, "n": 2},
    )
    plan, stats = build_color_plan(topo, config)
    validation = validate_plan(plan, topo, routes)
    assert validation.edge_conflicts == []
    assert validation.missing_logical == set()
    assert stats.C >= plan.color_lower_bound


@pytest.mark.ortools
def test_ilp_not_worse_than_greedy_on_k2_n4() -> None:
    pytest.importorskip("ortools")
    topo = _topo_k2_n4()
    greedy_cfg = ColorPlannerConfig(
        planner=PLANNER_GREEDY_FIRST_FIT,
        link_universe=LINK_UNIVERSE_FULL,
        topology_hint={"k": 2, "n": 4},
    )
    ilp_cfg = ColorPlannerConfig(
        planner=PLANNER_ILP_MIN_C,
        link_universe=LINK_UNIVERSE_FULL,
        topology_hint={"k": 2, "n": 4},
        time_limit_s=30.0,
    )
    greedy_plan, _ = build_color_plan(topo, greedy_cfg)
    ilp_plan, stats = build_color_plan(topo, ilp_cfg)
    assert ilp_plan.C <= greedy_plan.C
    assert not stats.ilp_fallback_used


def test_topology_coloring_strategy_hook() -> None:
    topo = TDMFlatButterfly(
        k=2,
        n=4,
        rows=4,
        cols=4,
        color_planner_config=ColorPlannerConfig(
            planner=PLANNER_GREEDY_FIRST_FIT,
            topology_hint={"k": 2, "n": 4},
        ),
    )
    plan = topo.coloring()
    assert plan.C >= 2
    stats = plan_stats(plan)
    assert stats.mean_links_per_color > 0

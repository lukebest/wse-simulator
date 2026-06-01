"""Tests for color visualization schedule builder."""

from __future__ import annotations

import pytest

from wsesim.network.color_catalog import (
    C_AG_COL_NORTH,
    C_AG_COL_SOUTH,
    C_AG_ROW_EAST,
    C_AG_ROW_WEST,
    C_BCAST_COL_SOUTH,
    C_BCAST_ROW_EAST,
    build_ideal_plan,
)
from wsesim.network.color_usage import ColorBudget
from wsesim.network.color_viz import (
    build_baseline_schedule,
    build_viz_schedule,
    expand_path,
    summarize_budgets,
    traffic_to_flows,
)
from wsesim.network.collective import generate_ideal_collective


def test_expand_path_bus_broadcast_row():
    plan = build_ideal_plan(4, 4)
    hops = expand_path(plan, 0, 1, C_BCAST_ROW_EAST)
    assert hops == [(0, 1)]


def test_expand_path_bus_multi_hop_column():
    plan = build_ideal_plan(4, 4)
    hops = expand_path(plan, 0, 8, C_BCAST_COL_SOUTH)
    assert hops == [(0, 4), (4, 8)]


def test_expand_path_unicast_xy():
    plan = build_ideal_plan(4, 4)
    hops = expand_path(plan, 0, 15, 0)
    assert hops == [(0, 4), (4, 8), (8, 12), (12, 13), (13, 14), (14, 15)]


def test_traffic_to_flows_has_vn_stream():
    plan = build_ideal_plan(4, 4)
    traffic = generate_ideal_collective("broadcast", 4, 4, 128)
    flows = traffic_to_flows(traffic, plan, flits=2)
    assert flows
    assert flows[0].vn_stream.startswith("c")
    assert len(flows[0].path) >= 1


@pytest.mark.parametrize("budget", list(ColorBudget))
def test_distinct_colors_within_budget(budget):
    for pattern in ("broadcast", "gather", "reduce", "allgather", "allreduce"):
        traffic = generate_ideal_collective(
            pattern, 4, 4, 128, color_budget=budget
        )
        sched = build_viz_schedule(4, 4, pattern, budget=budget, flits=1)
        assert sched.distinct_colors <= {
            ColorBudget.MINIMAL: 1,
            ColorBudget.COMPACT: 2,
            ColorBudget.PARALLEL: 4,
        }[budget], (pattern, budget, sched.distinct_colors)


def test_minimal_allgather_one_color():
    sched = build_viz_schedule(4, 4, "allgather", budget=ColorBudget.MINIMAL, flits=1)
    assert sched.distinct_colors == 1


def test_parallel_allgather_four_colors():
    sched = build_viz_schedule(4, 4, "allgather", budget=ColorBudget.PARALLEL, flits=1)
    assert sched.distinct_colors == 4
    used = {f.color_id for f in sched.flows}
    assert used == {C_AG_ROW_EAST, C_AG_ROW_WEST, C_AG_COL_SOUTH, C_AG_COL_NORTH}


def test_parallel_broadcast_two_bus_colors():
    sched = build_viz_schedule(4, 4, "broadcast", budget=ColorBudget.PARALLEL, flits=1, root=0)
    assert sched.distinct_colors == 2
    used = {f.color_id for f in sched.flows}
    assert C_BCAST_ROW_EAST in used
    assert C_BCAST_COL_SOUTH in used


def test_summarize_budgets_four_schemes():
    rows = summarize_budgets(4, 4, "allreduce", flits=3)
    assert len(rows) == 4
    schemes = {r["scheme"] for r in rows}
    assert schemes == {"xy_baseline", "color_minimal", "color_compact", "color_parallel"}


@pytest.mark.parametrize("mesh", [(4, 4), (8, 8)])
def test_schedule_duration_positive(mesh):
    rows, cols = mesh
    for pattern in ("broadcast", "allgather", "allreduce"):
        sched = build_viz_schedule(rows, cols, pattern, budget=ColorBudget.PARALLEL, flits=3)
        assert sched.duration > 0
        assert sched.peak >= 1


def test_baseline_single_color():
    sched = build_baseline_schedule(4, 4, "broadcast", flits=1)
    assert sched.distinct_colors == 1
    assert sched.budget == "xy_baseline"

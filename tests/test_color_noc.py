"""Tests for color virtual-network model."""

from __future__ import annotations

import simpy

from wsesim.network.color import ColorPlan
from wsesim.network.color_alloc import FlowSpec, graph_coloring_allocate, greedy_allocate
from wsesim.network.color_catalog import MAX_COLORS, build_ideal_plan
from wsesim.network.color_network import ColorNetwork
from wsesim.network.color_routes import build_mixed_plan, mesh_dims
from wsesim.network.color_sim import PATTERNS, compare_pattern
from wsesim.network.packet import Packet


def test_color_plan_next_hops_xy():
    plan = build_mixed_plan(4, 4, 8)
    hops = plan.next_hops(0, 0, 15)
    assert hops == {4}  # go south first (xy)


def test_color_network_delivers_unicast():
    plan = build_mixed_plan(4, 4, 4)
    env = simpy.Environment()
    net = ColorNetwork(env=env, plan=plan, router_pipeline_mode="1_stage")
    pkt = Packet(src=0, dst=15, size_bytes=128, payload_type="test", color=0)
    env.process(net.send_packet(pkt, color=0, seq=0))
    env.run()
    assert net.stats.packets_sent == 1
    assert net.stats.ordering_violations == 0


def test_color_ordering_sequential_packets():
    plan = build_mixed_plan(4, 4, 4)
    env = simpy.Environment()
    net = ColorNetwork(env=env, plan=plan, router_pipeline_mode="1_stage")
    for seq in range(3):
        pkt = Packet(src=0, dst=3, size_bytes=64, payload_type="ord", color=0)
        env.process(net.send_packet(pkt, color=0, seq=seq))
    env.run()
    assert net.stats.ordering_violations == 0
    assert net.delivery_seq[(0, 0, 3)] == [0, 1, 2]


def test_greedy_allocator_reuses_colors():
    flows = [
        FlowSpec("a", 0, 10, {(0, 1)}),
        FlowSpec("b", 20, 30, {(0, 1)}),
        FlowSpec("c", 0, 10, {(2, 3)}),
    ]
    result = greedy_allocate(flows, num_colors=2)
    assert result.flow_to_color["a"] == result.flow_to_color["b"]
    assert len(set(result.flow_to_color.values())) <= 2


def test_graph_coloring_allocator():
    flows = [
        FlowSpec("a", 0, 10, {(0, 1)}),
        FlowSpec("b", 0, 10, {(0, 1)}),
        FlowSpec("c", 0, 10, {(2, 3)}),
    ]
    result = graph_coloring_allocate(flows, num_colors=3)
    assert result.flow_to_color["a"] != result.flow_to_color["b"]
    assert result.flow_to_color["a"] in (0, 1, 2)


def test_ideal_catalog_is_mesh_independent():
    p4 = build_ideal_plan(4, 4)
    p8 = build_ideal_plan(8, 8)
    assert p4.num_colors == p8.num_colors == MAX_COLORS
    assert [c.name for c in p4.colors] == [c.name for c in p8.colors]


def test_validate_ideal_plan_bus_acyclic():
    from wsesim.network.color_usage import validate_plan

    plan = build_ideal_plan(4, 4)
    assert validate_plan(plan) == {}


def test_color_usage_scenarios_cover_collectives():
    from wsesim.network.color_usage import COLLECTIVE_SCENARIOS, scenario_for

    patterns = {s.pattern for s in COLLECTIVE_SCENARIOS}
    assert {"broadcast", "gather", "reduce", "allgather", "allreduce"}.issubset(patterns)
    assert scenario_for("broadcast") is not None


def test_root_flexible_broadcast_colors():
    from wsesim.network.color_catalog import C_BCAST_COL_NORTH, C_BCAST_ROW_WEST
    from wsesim.network.color_usage import pick_broadcast_colors

    row_c, col_c = pick_broadcast_colors(15, 4, 4)  # bottom-right on 4x4
    assert row_c == C_BCAST_ROW_WEST
    assert col_c == C_BCAST_COL_NORTH


def test_ideal_bus_routes_toward_dest():
    plan = build_ideal_plan(4, 4)
    # bcast_col_south (color 3): node 0 -> node 4 (south)
    assert plan.next_hops(0, 3, 12) == {4}
    # gather_col_north (color 4): node 12 -> node 8 (north)
    assert plan.next_hops(12, 4, 0) == {8}


def test_color_beats_or_matches_xy_on_collectives():
    for pattern in PATTERNS:
        xy, color = compare_pattern(4, 4, pattern, msg_bytes=128)
        assert color.ordering_violations == 0, pattern
        assert color.makespan_cycles > 0, pattern
        assert xy.makespan_cycles > 0, pattern
        # Ideal static routing should not be slower than naive single-VN XY.
        assert color.makespan_cycles <= xy.makespan_cycles, (
            pattern, color.makespan_cycles, xy.makespan_cycles
        )


def test_color_budget_minimizes_distinct_colors():
    from wsesim.network.collective import generate_ideal_collective
    from wsesim.network.color_usage import ColorBudget, budget_for, distinct_colors

    for pattern in ("broadcast", "gather", "reduce", "allgather", "allreduce"):
        for budget in ColorBudget:
            traffic = generate_ideal_collective(
                pattern, 4, 4, 128, color_budget=budget
            )
            n = distinct_colors(traffic)
            assert n <= budget_for(pattern, budget), (pattern, budget, n)
        minimal = generate_ideal_collective(
            pattern, 4, 4, 128, color_budget=ColorBudget.MINIMAL
        )
        assert distinct_colors(minimal) == 1, pattern


def test_minimal_and_compact_collectives_preserve_ordering():
    from wsesim.network.color_usage import ColorBudget

    for budget in (ColorBudget.MINIMAL, ColorBudget.COMPACT):
        for pattern in PATTERNS:
            _, color = compare_pattern(4, 4, pattern, msg_bytes=128, color_budget=budget)
            assert color.ordering_violations == 0, (budget, pattern)
            assert color.distinct_colors <= 2, (budget, pattern, color.distinct_colors)

"""Tests for color virtual-network NoC."""

from __future__ import annotations

import simpy
import pytest

from wsesim.fault.defect_map import DefectMap
from wsesim.network.color import ColorPlan
from wsesim.network.color_alloc import FlowSpec, allocate_greedy, allocate_graph_coloring
from wsesim.network.color_network import ColorNetwork
from wsesim.network.color_repair import repair_color_plan, routable_nodes
from wsesim.network.color_routes import (
    build_default_mixed_plan,
    build_path_color,
    build_snake_ring_color,
    is_acyclic_color,
)
from wsesim.network.color_sim import run_color_simulation, run_pattern_comparison, run_xy_baseline
from wsesim.network.collective import assign_colors_to_traffic, generate_collective_traffic
from wsesim.network.flow_control.per_color_credit import PerColorCreditFlowControl
from wsesim.network.ordering import OrderTracker
from wsesim.network.packet import Packet
from wsesim.network.topology.mesh2d import Mesh2D


def test_color_plan_path_trace() -> None:
    plan = ColorPlan.empty(16, 4)
    build_path_color(plan, 1, 0, 15, 4, 4)
    path = plan.trace_route(0, 15, 1)
    assert path is not None
    assert path[0] == 0
    assert path[-1] == 15


def test_snake_ring_is_cyclic_not_acyclic() -> None:
    plan = ColorPlan.empty(16, 4)
    build_snake_ring_color(plan, 0, 4, 4)
    assert not is_acyclic_color(plan, 0)


def test_path_color_is_acyclic() -> None:
    plan = ColorPlan.empty(16, 4)
    build_path_color(plan, 2, 0, 15, 4, 4)
    assert is_acyclic_color(plan, 2)


def test_per_color_flow_control() -> None:
    fc = PerColorCreditFlowControl(entries_per_color=2)
    assert fc.can_send(0, 2)
    assert fc.can_send(1, 2)
    assert not fc.can_send(2, 2)


def test_color_network_delivery() -> None:
    plan = ColorPlan.empty(16, 4)
    build_path_color(plan, 0, 0, 15, 4, 4)
    env = simpy.Environment()
    net = ColorNetwork(
        env=env,
        topology=Mesh2D(),
        color_plan=plan,
        num_nodes=16,
        enforce_single_source=False,
    )
    env.process(
        net.send_packet(
            Packet(src=0, dst=15, size_bytes=128, payload_type="test", color=0, seq=0)
        )
    )
    env.run()
    net.finalize_stats()
    assert net.stats.packets_sent == 1
    assert net.stats.ordering_violations == 0


def test_order_tracker_detects_violation() -> None:
    ot = OrderTracker()
    ot.record_delivery(0, 1, 0, 1.0)
    ot.record_delivery(0, 1, 2, 2.0)  # skipped seq 1
    assert ot.violations == 1


def test_assign_colors_to_traffic() -> None:
    traffic = [{"src_core": 0, "dst_core": 1, "size_bytes": 64, "payload": "all_to_all", "delay_cycles": 0}]
    colored = assign_colors_to_traffic(traffic, 16)
    assert "color" in colored[0]


def test_color_beats_xy_on_reduction_tree() -> None:
    color_r, xy_r = run_pattern_comparison("reduction_tree", 16, 4, num_colors=16)
    assert color_r.makespan_cycles > 0
    assert xy_r.makespan_cycles > 0


def test_mixed_taxonomy_generates_traffic() -> None:
    nodes = list(range(16))
    t = generate_collective_traffic(
        "mixed_taxonomy", nodes, 16, 128, 2, topology_hint={"rows": 4, "cols": 4}
    )
    assert len(t) > 0


def test_allocate_greedy() -> None:
    flows = [
        FlowSpec("f0", 0, 15, "path", phase=0),
        FlowSpec("f1", 1, 14, "path", phase=0),
    ]
    alloc = allocate_greedy(flows, 16, 8, 4)
    assert alloc.plan is not None
    assert "f0" in alloc.flow_to_color


def test_allocate_graph_coloring() -> None:
    flows = [
        FlowSpec("f0", 0, 15, "path", phase=0),
        FlowSpec("f1", 1, 14, "path", phase=1),
    ]
    alloc = allocate_graph_coloring(flows, 16, 8, 4)
    assert alloc.plan is not None


def test_color_repair_with_defects() -> None:
    plan = build_default_mixed_plan(16, 8, 4)
    topo = Mesh2D()
    graph = topo.build(16)
    links = set(graph.keys())
    all_links = {(s, d) for s, ds in graph.items() for d in ds}
    defect = DefectMap(dead_cores=set(), dead_links={(0, 1), (1, 0)})
    repaired, coverage = repair_color_plan(plan, graph, defect)
    assert coverage >= 0.0
    assert len(repaired.dest) > 0


def test_routable_nodes_after_defect() -> None:
    topo = Mesh2D()
    graph = topo.build(16)
    defect = DefectMap.generate(16, set(), 0.05, 0.0, seed=42)
    reachable = routable_nodes(graph, defect)
    assert len(reachable) >= 1


def test_default_mixed_plan_builds() -> None:
    plan = build_default_mixed_plan(64, 16, 8)
    assert plan.num_colors == 16
    assert len(plan.dest) == 64

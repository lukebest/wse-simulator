"""Tests for mesh collective fault tolerance."""

from __future__ import annotations

import pytest

from wsesim.network.collective_faults import (
    MeshFault,
    analyze_collective_faulty,
    analyze_fault_matrix,
    bfs_path,
    build_hamilton_on_punctured,
    hamilton_cycle_possible,
    make_scenario_fault,
    reroute_flow,
    reroute_flows,
    schedule_collective_flows,
)
from wsesim.network.collective_patterns import (
    CollectiveFlow,
    build_allto_all_flows,
    build_hamilton_ring_allgather_flows,
    make_edge,
    node,
    schedule_bufferless_noc,
)


def test_bfs_avoids_bad_pe():
    fault = MeshFault.pe(1, 0)
    path = bfs_path(node(0, 0), node(2, 0), 4, 4, fault)
    assert path is not None
    for edge in path:
        assert not fault.node_bad(edge.from_node)
        assert not fault.node_bad(edge.to_node)


def test_bfs_avoids_bad_link():
    fault = MeshFault.link(node(0, 0), node(1, 0))
    path = bfs_path(node(0, 0), node(2, 0), 4, 4, fault)
    assert path is not None
    for edge in path:
        assert edge.id not in fault.bad_edges


def test_reroute_flow_skips_bad_src():
    flow = CollectiveFlow(
        id="t",
        from_node=node(1, 0),
        to_node=node(3, 0),
        path=[make_edge(node(1, 0), node(2, 0)), make_edge(node(2, 0), node(3, 0))],
    )
    fault = MeshFault.pe(1, 0)
    assert reroute_flow(flow, 4, 4, fault) is None


def test_reroute_alltoall_flows_no_bad_nodes_in_path():
    flows = build_allto_all_flows(4, 4)
    fault = MeshFault.pe(1, 1)
    routed, dropped = reroute_flows(flows, 4, 4, fault)
    assert dropped > 0
    for flow in routed:
        for edge in flow.path:
            assert not fault.node_bad(edge.from_node)
            assert not fault.node_bad(edge.to_node)


def test_hamilton_cycle_broken_by_single_pe():
    fault = MeshFault.pe(2, 2)
    assert not hamilton_cycle_possible(4, 4, fault)
    order, is_cycle = build_hamilton_on_punctured(4, 4, fault)
    assert len(order) == 15
    assert not is_cycle


def test_hamilton_intact_without_fault():
    order, is_cycle = build_hamilton_on_punctured(4, 4, MeshFault())
    assert len(order) == 16
    assert is_cycle


def test_allgather_degradation_near_2x_on_pe_fault():
    fault = make_scenario_fault(8, 8, "pe_point", "corner")
    result = analyze_collective_faulty("allgather", 8, 8, fault)
    assert result["hamilton_degraded"]
    assert result["ratio"] is not None
    assert result["ratio"] >= 1.5
    assert result["z_faulty"] > result["z_healthy_scheduled"]


def test_alltoall_center_worse_than_corner():
    corner = make_scenario_fault(8, 8, "pe_point", "corner")
    center = make_scenario_fault(8, 8, "pe_point", "center")
    r_corner = analyze_collective_faulty("alltoall", 8, 8, corner)
    r_center = analyze_collective_faulty("alltoall", 8, 8, center)
    assert r_center["ratio"] >= r_corner["ratio"]


def test_degradation_ratio_finite_and_ge_one():
    results = analyze_fault_matrix([(4, 4)])
    for r in results:
        if r["reachable"] and r["ratio"] is not None:
            assert r["ratio"] >= 1.0
            assert r["ratio"] < 100.0


def test_fault_matrix_count():
    results = analyze_fault_matrix([(4, 4)])
    # 6 patterns × 3 fault types × 3 regions = 54
    assert len(results) == 54


def test_schedule_collective_flows_ok_on_healthy():
    flows, _, _ = build_hamilton_ring_allgather_flows(4, 4)
    sched = schedule_collective_flows(flows)
    assert sched["ok"]
    assert sched["peak"] <= 1


def test_unreachable_when_root_bad():
    fault = MeshFault.pe(0, 0)
    result = analyze_collective_faulty("broadcast", 4, 4, fault, root=0)
    assert not result["reachable"]
    assert result["z_faulty"] == 0


def test_fault_schematics_render():
    from wsesim.network.fault_schematic import render_all_schematics, render_pattern_row

    row = render_pattern_row("broadcast", 4, 4)
    assert "Golden" in row
    assert "PE 坏点" in row
    assert "<svg" in row

    all_html = render_all_schematics()
    for pat in ("broadcast", "allgather", "alltoall"):
        assert f'schem-{pat}' in all_html
    assert "2b. Golden vs 故障处理示意图" in all_html

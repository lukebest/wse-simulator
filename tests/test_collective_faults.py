"""Tests for mesh collective fault tolerance."""

from __future__ import annotations

import pytest

from wsesim.network.collective_faults import (
    MeshFault,
    analyze_collective_faulty,
    analyze_fault_matrix,
    analyze_fault_matrix_topology,
    bfs_path,
    bipartite_survivor_counts,
    build_hamilton_on_punctured,
    compile_fault_study_8x8,
    hamilton_cycle_possible,
    is_enhanced_perimeter_link,
    make_scenario_fault,
    reroute_flow,
    reroute_flows,
    schedule_collective_flows,
    supermesh_bi_enhanced_pairs,
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


def test_block1x2_removes_two_adjacent_and_keeps_parity():
    fault = make_scenario_fault(8, 8, "pe_block_1x2", "corner")
    assert fault.bad_nodes == frozenset({(0, 0), (1, 0)})
    # 1×2 removes one even + one odd node → survivor color classes stay balanced
    even, odd = bipartite_survivor_counts(8, 8, fault)
    assert even == odd


def test_block1x2_alltoall_between_point_and_2x2():
    point = analyze_collective_faulty(
        "alltoall", 8, 8, make_scenario_fault(8, 8, "pe_point", "center")
    )["ratio"]
    b1x2 = analyze_collective_faulty(
        "alltoall", 8, 8, make_scenario_fault(8, 8, "pe_block_1x2", "center")
    )["ratio"]
    assert b1x2 >= point


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


def test_supermesh_corner_link_is_enhanced_perimeter():
    fault = make_scenario_fault(8, 8, "link_point", "corner")
    assert is_enhanced_perimeter_link(8, 8, fault)


def test_supermesh_8x8_study_compiles():
    study = compile_fault_study_8x8()
    assert study["rows"] == 8
    assert len(study["comparison"]) == 54
    assert study["healthy_bounds_supermesh_bi"]["alltoall"] < study["healthy_bounds_mesh"]["alltoall"]


def test_supermesh_alltoall_healthy_makespan_le_mesh():
    mesh = analyze_fault_matrix_topology([(8, 8)], topology="mesh")
    sm = analyze_fault_matrix_topology([(8, 8)], topology="supermesh_bi")
    m_a2a = next(
        r for r in mesh
        if r["pattern"] == "alltoall" and r["fault_type"] == "link_point" and r["region"] == "corner"
    )
    s_a2a = next(
        r for r in sm
        if r["pattern"] == "alltoall" and r["fault_type"] == "link_point" and r["region"] == "corner"
    )
    assert s_a2a["z_healthy_scheduled"] <= m_a2a["z_healthy_scheduled"]

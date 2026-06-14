"""Verify Hamiltonian-ring allgather vs dimwise on 4x4, 8x8, 12x16 meshes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from wsesim.network.collective_patterns import (
    build_alltoall_twophase_flows,
    build_hamilton_cycle,
    build_hamilton_ring_allgather_flows,
    build_hamilton_ring_allgather_flows_aniso,
    compare_allgather_patterns,
    compile_case_study_8x8_report,
    dilate_slots,
    effective_pe_io_rate,
    enhanced_edge_id_set,
    optimal_hamilton_makespan,
    optimal_z_lower_bound,
    optimal_z_lower_bound_aniso,
    optimal_z_lower_bound_bw,
    optimal_z_lower_bound_h,
    schedule_bufferless_noc,
    verify_edge_capacity,
    verify_hop_spacing,
    verify_hop_spacing_aniso,
    verify_port_bandwidth,
    verify_preassigned_slots,
)
from wsesim.network.topology.supermesh_alter import SuperMeshAlter
from wsesim.network.topology.supermesh_bi import SuperMeshBi


MESHES = [(4, 4), (8, 8), (12, 16)]


@pytest.mark.parametrize("rows,cols", MESHES)
def test_hamilton_cycle_covers_all_nodes(rows: int, cols: int) -> None:
    cycle = build_hamilton_cycle(rows, cols)
    assert len(cycle) == rows * cols
    assert len(set((n.x, n.y) for n in cycle)) == rows * cols


@pytest.mark.parametrize("rows,cols", MESHES)
def test_hamilton_allgather_makespan_and_zero_stall(rows: int, cols: int) -> None:
    n = rows * cols
    t_star = optimal_hamilton_makespan(n)
    flows, expected_t, meta = build_hamilton_ring_allgather_flows(rows, cols, 1)
    result = verify_preassigned_slots(flows)

    assert meta["t_star"] == t_star
    assert expected_t == t_star
    assert result.ok, result.collision
    assert result.total_stall == 0
    assert result.makespan == t_star
    assert result.makespan == math.ceil((n - 1) / 2)


@pytest.mark.parametrize("rows,cols", MESHES)
def test_hamilton_allgather_peak_link_utilization(rows: int, cols: int) -> None:
    flows, _, _ = build_hamilton_ring_allgather_flows(rows, cols, 1)
    result = verify_preassigned_slots(flows)
    assert result.peak == 1


@pytest.mark.parametrize("rows,cols", MESHES)
def test_compare_hamilton_beats_dimwise(rows: int, cols: int) -> None:
    report = compare_allgather_patterns(rows, cols, 1)
    assert report["hamilton"]["ok"]
    assert report["hamilton"]["stall"] == 0
    assert report["hamilton"]["makespan"] == report["t_star"]
    assert report["dimwise"]["ok"]
    assert report["hamilton"]["makespan"] < report["dimwise"]["makespan"]


def test_hamilton_requires_even_cols() -> None:
    with pytest.raises(ValueError, match="even cols"):
        build_hamilton_cycle(3, 5)


@pytest.mark.parametrize("rows,cols", [(4, 4), (8, 8)])
def test_alltoall_twophase_conflict_free_zero_stall(rows: int, cols: int) -> None:
    flows = build_alltoall_twophase_flows(rows, cols)
    result = verify_preassigned_slots(flows)
    z_star = optimal_z_lower_bound("alltoall", rows, cols)
    assert result.ok, result.collision
    assert result.peak == 1
    assert result.total_stall == 0
    assert result.makespan >= z_star  # cannot beat the bisection lower bound
    assert result.makespan <= 3 * z_star  # 2-phase stays within a small constant


@pytest.mark.parametrize("rows,cols", [(4, 4), (8, 8)])
@pytest.mark.parametrize("h", [2, 4])
def test_alltoall_twophase_multicycle_links(rows: int, cols: int, h: int) -> None:
    """Strided two-phase under hop latency h: conflict-free, h-spaced, near-additive cost."""
    base = verify_preassigned_slots(build_alltoall_twophase_flows(rows, cols))
    flows = build_alltoall_twophase_flows(rows, cols, hop_latency=h)
    result = verify_preassigned_slots(flows)
    assert result.ok, result.collision
    assert result.peak == 1
    assert verify_hop_spacing(flows, h)
    # bandwidth term is untouched: cost grows additively (O(h*(cols+rows))),
    # nowhere near the multiplicative h*Z_1 of naive dilation.
    assert result.makespan < base.makespan + h * (cols + rows) + h * 10
    assert result.makespan + h >= optimal_z_lower_bound_h(
        "alltoall", rows, cols, hop_latency=h
    )


@pytest.mark.parametrize("rows,cols", [(4, 4), (8, 8)])
def test_dilation_preserves_conflict_freedom(rows: int, cols: int) -> None:
    """slot -> h*slot keeps (edge, slot) uniqueness and enforces h-spacing."""
    h = 4
    flows, t_star, _ = build_hamilton_ring_allgather_flows(rows, cols, 1)
    dilate_slots(flows, h)
    result = verify_preassigned_slots(flows)
    assert result.ok, result.collision
    assert result.peak == 1
    assert verify_hop_spacing(flows, h)
    assert result.makespan == h * (t_star - 1) + 1  # last launch; arrival = h*t_star


@pytest.mark.parametrize("rows,cols", [(4, 4), (8, 8), (12, 16)])
def test_hamilton_allgather_aniso_conflict_free(rows: int, cols: int) -> None:
    """Cumulative-weighted ring timing under (hx=4, hy=8): conflict-free, h-spaced."""
    hx, hy = 4, 8
    flows, arrival, meta = build_hamilton_ring_allgather_flows_aniso(
        rows, cols, hx, hy, orient="auto"
    )
    result = verify_preassigned_slots(flows)
    assert result.ok, result.collision
    assert result.peak == 1
    assert verify_hop_spacing_aniso(flows, hx, hy)
    # hy > hx: the horizontal-heavy row snake must win the orientation choice.
    assert meta["orient"] == "row"
    assert arrival >= optimal_z_lower_bound_aniso(
        "allgather", rows, cols, hx=hx, hy=hy
    )


def test_hamilton_allgather_aniso_orientation_matters() -> None:
    hx, hy = 4, 8
    _, z_col, _ = build_hamilton_ring_allgather_flows_aniso(8, 8, hx, hy, orient="col")
    _, z_row, _ = build_hamilton_ring_allgather_flows_aniso(8, 8, hx, hy, orient="row")
    assert z_row < z_col  # snake along the cheap axis wins when hy > hx


@pytest.mark.parametrize("rows,cols", [(4, 4), (8, 8)])
def test_alltoall_twophase_aniso_links(rows: int, cols: int) -> None:
    """Per-axis strides (hx=4, hy=8): conflict-free, spaced, additive overhead."""
    hx, hy = 4, 8
    base = verify_preassigned_slots(build_alltoall_twophase_flows(rows, cols))
    flows = build_alltoall_twophase_flows(rows, cols, hop_latency_x=hx, hop_latency_y=hy)
    result = verify_preassigned_slots(flows)
    assert result.ok, result.collision
    assert result.peak == 1
    assert verify_hop_spacing_aniso(flows, hx, hy)
    assert result.makespan < base.makespan + hx * cols + hy * rows + hy * 10
    assert result.makespan + hy >= optimal_z_lower_bound_aniso(
        "alltoall", rows, cols, hx=hx, hy=hy
    )


HX_CASE, HY_CASE = 4, 8
ROWS_CASE, COLS_CASE = 8, 8


@pytest.mark.parametrize("inject_bw", [1, 2, 4])
def test_8x8_bounds_invariant_in_inject_bw(inject_bw: int) -> None:
    """At 8x8 hx=4/hy=8, latency/bisection dominate -> z* flat in b."""
    expected = {
        "broadcast": 84,
        "reduce": 84,
        "allreduce": 84,
        "allgather": 84,
        "gather": 84,
        "alltoall": 128,
    }
    for pattern, z in expected.items():
        got = optimal_z_lower_bound_bw(
            pattern, ROWS_CASE, COLS_CASE, hx=HX_CASE, hy=HY_CASE, inject_bw=inject_bw
        )
        assert got == z, f"{pattern} b={inject_bw}"


def test_8x8_allgather_port_bandwidth_by_scheme() -> None:
    bidir, _, _ = build_hamilton_ring_allgather_flows_aniso(
        ROWS_CASE, COLS_CASE, HX_CASE, HY_CASE, orient="row"
    )
    ok1, _ = verify_port_bandwidth(bidir, ROWS_CASE, COLS_CASE, 1, HX_CASE, HY_CASE)
    ok2, _ = verify_port_bandwidth(bidir, ROWS_CASE, COLS_CASE, 2, HX_CASE, HY_CASE)
    assert not ok1
    assert ok2

    stag, mk_stag, meta = build_hamilton_ring_allgather_flows_aniso(
        ROWS_CASE, COLS_CASE, HX_CASE, HY_CASE, orient="row", stagger_ccw=True
    )
    ok_stag, _ = verify_port_bandwidth(stag, ROWS_CASE, COLS_CASE, 1, HX_CASE, HY_CASE)
    assert verify_preassigned_slots(stag).ok
    assert verify_hop_spacing_aniso(stag, HX_CASE, HY_CASE)
    assert ok_stag
    assert mk_stag > 168
    assert meta["ccw_time_offset"] == optimal_hamilton_makespan(64)


def test_8x8_alltoall_twophase_port_and_supermesh() -> None:
    flows = build_alltoall_twophase_flows(
        ROWS_CASE, COLS_CASE, hop_latency_x=HX_CASE, hop_latency_y=HY_CASE
    )
    assert verify_preassigned_slots(flows).ok
    assert not verify_port_bandwidth(flows, ROWS_CASE, COLS_CASE, 1, HX_CASE, HY_CASE)[0]
    assert verify_port_bandwidth(flows, ROWS_CASE, COLS_CASE, 2, HX_CASE, HY_CASE)[0]

    bi = SuperMeshBi(rows=ROWS_CASE, cols=COLS_CASE).enhanced_edges(64)
    assert optimal_z_lower_bound_bw(
        "alltoall", ROWS_CASE, COLS_CASE, hx=HX_CASE, hy=HY_CASE,
        inject_bw=1, enhanced_undirected=bi,
    ) == 103

    alter = SuperMeshAlter(rows=ROWS_CASE, cols=COLS_CASE).enhanced_edges(64)
    assert optimal_z_lower_bound_bw(
        "alltoall", ROWS_CASE, COLS_CASE, hx=HX_CASE, hy=HY_CASE,
        inject_bw=1, enhanced_undirected=alter,
    ) == 128

    cap = {e: 2 for e in enhanced_edge_id_set(ROWS_CASE, COLS_CASE, bi)}
    a2a_bi = build_alltoall_twophase_flows(
        ROWS_CASE, COLS_CASE, hop_latency_x=HX_CASE, hop_latency_y=HY_CASE,
        enhanced_undirected=bi,
    )
    ok_e, _, peak = verify_edge_capacity(a2a_bi, cap)
    assert ok_e
    assert peak == 2


def test_compile_case_study_8x8_report() -> None:
    bi = SuperMeshBi(rows=8, cols=8).enhanced_edges(64)
    alter = SuperMeshAlter(rows=8, cols=8).enhanced_edges(64)
    report = compile_case_study_8x8_report(
        hx=HX_CASE, hy=HY_CASE, enhanced_bi=bi, enhanced_alter=alter
    )
    assert report["mesh"]["b1"]["bounds"]["alltoall"] == 128
    assert report["supermesh_bi"]["bounds_alltoall"] == 103
    assert report["supermesh_alter"]["bounds_alltoall"] == 128
    assert report["mesh"]["schemes"]["allgather_bidir"]["port_bw"]["b1"] is False
    assert report["mesh"]["schemes"]["allgather_stagger_b1"]["port_bw"]["b1"] is True


def run_benchmark() -> list[dict]:
    return [compare_allgather_patterns(rows, cols, 1) for rows, cols in MESHES]


def write_results_json(path: Path) -> list[dict]:
    rows = run_benchmark()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    out = write_results_json(Path("outputs/hamilton_allgather/results.json"))
    for row in out:
        print(
            f"{row['rows']}x{row['cols']}: "
            f"hamilton makespan={row['hamilton']['makespan']} stall={row['hamilton']['stall']} | "
            f"dimwise makespan={row['dimwise']['makespan']} stall={row['dimwise']['stall']} | "
            f"speedup={row['speedup_vs_dimwise']:.2f}x"
        )

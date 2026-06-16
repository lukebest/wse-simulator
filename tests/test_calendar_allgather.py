"""Algorithm B (shortest-path backward calendar) allgather tests."""

from wsesim.network.collective_patterns import (
    build_hamilton_ring_allgather_flows_aniso,
    build_shortest_path_calendar_allgather_flows,
    compare_allgather_patterns,
    optimal_z_lower_bound_aniso,
    verify_hop_spacing_aniso,
    verify_preassigned_slots,
)


def _assert_calendar_optimal(rows: int, cols: int, hx: int = 4, hy: int = 4) -> None:
    lb = optimal_z_lower_bound_aniso("allgather", rows, cols, hx=hx, hy=hy)
    flows, mk, meta = build_shortest_path_calendar_allgather_flows(
        rows, cols, hop_latency_x=hx, hop_latency_y=hy
    )
    assert verify_preassigned_slots(flows).ok
    assert verify_hop_spacing_aniso(flows, hx, hy)
    assert mk == lb, f"makespan {mk} != lower bound {lb}, meta={meta}"
    assert meta["arrival_makespan"] == lb


def test_calendar_allgather_4x4_l4() -> None:
    _assert_calendar_optimal(4, 4)


def test_calendar_allgather_8x8_l4() -> None:
    _assert_calendar_optimal(8, 8)


def test_calendar_allgather_12x16_l4() -> None:
    _assert_calendar_optimal(12, 16)


def test_calendar_allgather_16x16_l4_bandwidth_bound() -> None:
    """16x16 is ingress-bound: T* = ceil((N-1)/2) = 128."""
    _assert_calendar_optimal(16, 16)


def test_calendar_beats_hamilton_aniso_8x8() -> None:
    hx = hy = 4
    _, cal_mk, _ = build_shortest_path_calendar_allgather_flows(
        8, 8, hop_latency_x=hx, hop_latency_y=hy
    )
    _, ham_mk, _ = build_hamilton_ring_allgather_flows_aniso(
        8, 8, hx, hy, orient="auto"
    )
    assert cal_mk < ham_mk
    assert cal_mk == 56
    assert ham_mk == 128


def test_compare_allgather_patterns_includes_calendar() -> None:
    report = compare_allgather_patterns(8, 8, hx=4, hy=4)
    assert report["calendar"]["ok"]
    assert report["calendar"]["makespan"] == 56
    assert report["calendar"]["optimal_aniso"]
    assert report["calendar_vs_hamilton_aniso"] > 2.0

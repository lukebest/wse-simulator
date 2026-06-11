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
    compare_allgather_patterns,
    optimal_hamilton_makespan,
    optimal_z_lower_bound,
    schedule_bufferless_noc,
    verify_preassigned_slots,
)


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

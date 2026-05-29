#!/usr/bin/env python3
"""Sweep defect rate on 4x4 color plan repair; write fault_degradation.csv."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from wsesim.fault.defect_map import DefectMap
from wsesim.network.color_alloc import build_mixed_workload_plan
from wsesim.network.color_repair import repair_color_plan, routable_nodes
from wsesim.network.color_routes import mesh_dims
from wsesim.network.topology.mesh2d import Mesh2D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="4x4")
    parser.add_argument("--output", type=Path, default=Path("outputs/color_vs_xy/fault"))
    parser.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows_s, cols_s = args.mesh.split("x")
    rows, cols = int(rows_s), int(cols_s)
    num_nodes = rows * cols
    args.output.mkdir(parents=True, exist_ok=True)

    plan = build_mixed_workload_plan(num_nodes, num_colors=16, cols=cols)
    topo = Mesh2D(rows=rows, cols=cols)
    graph = topo.build(num_nodes)

    out_rows: list[dict] = []
    rng = random.Random(args.seed)
    for rate in args.rates:
        n_dead = int(num_nodes * rate)
        dead = set(rng.sample(list(range(num_nodes)), n_dead)) if n_dead else set()
        defect = DefectMap(dead_cores=dead, dead_links=set())
        repaired, coverage = repair_color_plan(plan, graph, defect)
        routable = routable_nodes(graph, defect)
        out_rows.append(
            {
                "mesh": args.mesh,
                "defect_rate": rate,
                "dead_nodes": len(dead),
                "route_coverage": round(coverage, 4),
                "routable_fraction": round(len(routable) / num_nodes, 4),
            }
        )

    csv_path = args.output / "fault_degradation.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    (args.output / "fault_meta.json").write_text(
        json.dumps({"mesh": args.mesh, "num_colors": 16, "seed": args.seed}, indent=2)
    )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

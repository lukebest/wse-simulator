"""Simulation harness for color NoC vs baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import simpy

from wsesim.network.collective import assign_colors_to_traffic, generate_collective_traffic
from wsesim.network.color import ColorPlan
from wsesim.network.color_alloc import build_mixed_workload_plan
from wsesim.network.color_network import ColorNetwork
from wsesim.network.edge_routes import EdgeRouteTable, build_routes_for_traffic
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.topology.mesh2d import Mesh2D


@dataclass(slots=True)
class SimCaseResult:
    case: str
    topology: str
    algorithm: str
    mesh: str
    makespan_cycles: int
    avg_latency: float
    avg_link_util: float
    color_buffer_wait_cycles: int
    link_wait_cycles: int
    total_flits: int
    ordering_violations: int


PATTERNS = [
    "direct_allgather",
    "broadcast_tree",
    "reduction_tree",
    "all_to_all",
    "systolic",
    "mixed_taxonomy",
]

# Ring is included only for small meshes (high packet count)
PATTERNS_SMALL = ["ring"] + PATTERNS


def build_color_plan_for_traffic(
    traffic: list[dict],
    num_nodes: int,
    num_colors: int = 16,
    cols: int | None = None,
) -> tuple[ColorPlan, list[dict]]:
    """Build routes via EdgeRouteTable (delegates for compatibility)."""
    table, updated = build_routes_for_traffic(traffic, num_nodes, num_colors, cols)
    assert table.color_plan is not None
    return table.color_plan, updated


def run_color_simulation(
    traffic: list[dict],
    num_nodes: int,
    cols: int | None = None,
    num_colors: int = 16,
    msg_bytes: int = 128,
    link_bw: int = 1,
    enforce_single_source: bool = False,
) -> tuple[int, ColorNetwork]:
    """Run color NoC simulation; returns (makespan, network)."""
    if cols is not None:
        topology = Mesh2D(rows=num_nodes // cols, cols=cols)
    else:
        topology = Mesh2D()
    table, traffic = build_routes_for_traffic(traffic, num_nodes, num_colors, cols)
    env = simpy.Environment()
    net = ColorNetwork(
        env=env,
        topology=topology,
        color_plan=table.color_plan or ColorPlan.empty(num_nodes, num_colors),
        num_nodes=num_nodes,
        edge_routes=table,
        link_bw_flits_per_cycle=link_bw,
        entries_per_color=2,
        pipeline_cycles=1,
        flit_bytes=128,
        enforce_single_source=enforce_single_source,
    )

    def _send(pkt: dict):
        delay = int(pkt.get("delay_cycles", 0))
        if delay > 0:
            yield env.timeout(delay)
        yield env.process(
            net.send_packet(
                Packet(
                    src=int(pkt["src_core"]),
                    dst=int(pkt["dst_core"]),
                    size_bytes=int(pkt.get("size_bytes", msg_bytes)),
                    payload_type=str(pkt.get("payload", "data")),
                    color=int(pkt.get("color", 0)),
                    seq=int(pkt.get("seq", 0)),
                )
            )
        )

    for pkt in traffic:
        env.process(_send(pkt))
    env.run()
    net.finalize_stats()
    return int(env.now), net


def run_xy_baseline(
    traffic: list[dict],
    num_nodes: int,
    cols: int | None = None,
    msg_bytes: int = 128,
    link_bw: int = 1,
) -> tuple[int, UnifiedNetwork]:
    """Single-VN dimension-order XY baseline."""
    if cols is not None:
        topology = Mesh2D(rows=num_nodes // cols, cols=cols)
    else:
        topology = Mesh2D()
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=topology,
        routing=DimensionOrderRouting(),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=num_nodes,
        link_bw_flits_per_cycle=link_bw,
        link_latency_cycles=1,
        num_vcs=1,
        buffer_depth=8,
        router_pipeline_mode="1_stage",
        flit_bytes=128,
    )

    def _send(pkt: dict):
        delay = int(pkt.get("delay_cycles", 0))
        if delay > 0:
            yield env.timeout(delay)
        yield env.process(
            net.send_packet(
                Packet(
                    src=int(pkt["src_core"]),
                    dst=int(pkt["dst_core"]),
                    size_bytes=int(pkt.get("size_bytes", msg_bytes)),
                    payload_type=str(pkt.get("payload", "data")),
                )
            )
        )

    for pkt in traffic:
        env.process(_send(pkt))
    env.run()
    return int(env.now), net


def run_pattern_comparison(
    algorithm: str,
    num_nodes: int,
    cols: int | None = None,
    num_colors: int = 16,
    msg_bytes: int = 128,
    num_experts: int = 1,
) -> tuple[SimCaseResult, SimCaseResult]:
    """Compare color NoC vs XY baseline for one pattern."""
    # Ensure enough colors for unique static routes (one color per edge)
    num_colors = max(num_colors, min(256, num_nodes * num_nodes))
    nodes = list(range(num_nodes))
    hint = {"rows": num_nodes // cols, "cols": cols} if cols else {"rows": int(num_nodes**0.5), "cols": int(num_nodes**0.5)}
    raw = generate_collective_traffic(
        algorithm=algorithm,
        participating_nodes_global=nodes,
        cores_per_reticle=num_nodes,
        payload_bytes_per_expert=msg_bytes,
        num_experts=num_experts,
        topology_hint=hint,
    )
    color_traffic = assign_colors_to_traffic(raw, num_colors)

    color_ms, color_net = run_color_simulation(
        color_traffic, num_nodes, cols, num_colors, msg_bytes, enforce_single_source=False
    )
    xy_ms, xy_net = run_xy_baseline(raw, num_nodes, cols, msg_bytes)

    mesh_label = f"{hint['rows']}x{hint['cols']}"
    color_result = SimCaseResult(
        case=f"color_{algorithm}",
        topology="color_mesh2d",
        algorithm=algorithm,
        mesh=mesh_label,
        makespan_cycles=color_ms,
        avg_latency=color_net.stats.avg_latency(),
        avg_link_util=color_net.avg_link_utilization(),
        color_buffer_wait_cycles=color_net.stats.color_buffer_wait_cycles,
        link_wait_cycles=color_net.stats.link_wait_cycles,
        total_flits=color_net.stats.flits_sent,
        ordering_violations=color_net.stats.ordering_violations,
    )
    xy_result = SimCaseResult(
        case=f"xy_{algorithm}",
        topology="mesh2d_xy",
        algorithm=algorithm,
        mesh=mesh_label,
        makespan_cycles=xy_ms,
        avg_latency=xy_net.stats.avg_latency(),
        avg_link_util=0.0,
        color_buffer_wait_cycles=0,
        link_wait_cycles=xy_net.stats.link_wait_cycles,
        total_flits=xy_net.stats.flits_sent,
        ordering_violations=0,
    )
    return color_result, xy_result


def run_full_study(
    mesh_sizes: list[tuple[int, int]],
    output_dir: Path,
    num_colors: int = 16,
    msg_bytes: int = 128,
    write_trace: bool = False,
) -> list[dict]:
    """Run all patterns x mesh sizes; write results.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    for rows, cols in mesh_sizes:
        num_nodes = rows * cols
        patterns = PATTERNS_SMALL if num_nodes <= 16 else PATTERNS
        for pattern in patterns:
            print(f"  [{mesh_label}] {pattern}...", flush=True)
            color_r, xy_r = run_pattern_comparison(
                pattern, num_nodes, cols, num_colors, msg_bytes, num_experts=1
            )
            for r in (color_r, xy_r):
                row = {
                    "topology": r.topology,
                    "algorithm": r.algorithm,
                    "mesh": r.mesh,
                    "makespan_cycles": r.makespan_cycles,
                    "avg_latency": round(r.avg_latency, 2),
                    "avg_link_util": round(r.avg_link_util, 4),
                    "color_buffer_wait_cycles": r.color_buffer_wait_cycles,
                    "link_wait_cycles": r.link_wait_cycles,
                    "total_flits": r.total_flits,
                    "ordering_violations": r.ordering_violations,
                }
                rows_out.append(row)

    csv_path = output_dir / "results.csv"
    headers = list(rows_out[0].keys()) if rows_out else []
    with csv_path.open("w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows_out:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    summary = {
        "num_colors": num_colors,
        "msg_bytes": msg_bytes,
        "patterns": PATTERNS,
        "mesh_sizes": [f"{r}x{c}" for r, c in mesh_sizes],
        "color_wins": sum(
            1
            for i in range(0, len(rows_out), 2)
            if i + 1 < len(rows_out)
            and rows_out[i]["topology"] == "color_mesh2d"
            and rows_out[i]["makespan_cycles"] < rows_out[i + 1]["makespan_cycles"]
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if write_trace and mesh_sizes:
        rows, cols = mesh_sizes[0]
        num_nodes = rows * cols
        hint = {"rows": rows, "cols": cols}
        nodes = list(range(num_nodes))
        raw = generate_collective_traffic(
            "mixed_taxonomy", nodes, num_nodes, msg_bytes, 1, topology_hint=hint
        )
        table, traffic = build_routes_for_traffic(raw, num_nodes, num_colors, cols)
        if cols is not None:
            topology = Mesh2D(rows=rows, cols=cols)
        else:
            topology = Mesh2D()
        env = simpy.Environment()
        net = ColorNetwork(
            env=env,
            topology=topology,
            color_plan=table.color_plan or ColorPlan.empty(num_nodes, num_colors),
            num_nodes=num_nodes,
            edge_routes=table,
            enable_trace=True,
        )
        if traffic:
            pkt = traffic[0]

            def _one():
                yield env.process(
                    net.send_packet(
                        Packet(
                            src=int(pkt["src_core"]),
                            dst=int(pkt["dst_core"]),
                            size_bytes=msg_bytes,
                            payload_type=str(pkt.get("payload", "data")),
                            color=int(pkt["color"]),
                        )
                    )
                )

            env.process(_one())
            env.run()
        trace_path = output_dir / "cycle_trace.json"
        trace_path.write_text(json.dumps(net.trace[:500], indent=2))

    return rows_out

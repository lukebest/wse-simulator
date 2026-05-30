"""Simulation harness: color VN vs single-VN XY baseline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import simpy

from wsesim.fault.defect_map import DefectMap
from wsesim.network.collective import (
    IDEAL_PATTERNS,
    generate_baseline_collective,
    generate_ideal_collective,
)
from wsesim.network.color_catalog import build_baseline_plan, build_ideal_plan
from wsesim.network.color_network import ColorNetwork
from wsesim.network.color_repair import apply_defect_map_to_graph, repair_color_plan
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.topology.mesh2d import Mesh2D


PATTERNS = list(IDEAL_PATTERNS)


@dataclass(slots=True)
class SimCaseResult:
    mesh: str
    scheme: str
    pattern: str
    msg_bytes: int
    makespan_cycles: int
    avg_latency: float
    avg_link_util: float
    color_buffer_wait_cycles: int
    buffer_wait_cycles: int
    link_wait_cycles: int
    total_flits: int
    ordering_violations: int
    packets: int


def run_xy_baseline(
    rows: int,
    cols: int,
    traffic: list[dict],
    *,
    msg_bytes: int = 128,
) -> SimCaseResult:
    n = rows * cols
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=Mesh2D(rows=rows, cols=cols),
        routing=DimensionOrderRouting(),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=n,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=1,
        buffer_depth=8,
        router_pipeline_mode="1_stage",
        flit_bytes=msg_bytes,
    )
    _inject_traffic(env, net, traffic)
    env.run()
    return SimCaseResult(
        mesh=f"{rows}x{cols}",
        scheme="xy_single_vn",
        pattern="",
        msg_bytes=msg_bytes,
        makespan_cycles=int(env.now),
        avg_latency=net.stats.avg_latency(),
        avg_link_util=_xy_link_util(net),
        color_buffer_wait_cycles=0,
        buffer_wait_cycles=net.stats.buffer_wait_cycles,
        link_wait_cycles=net.stats.link_wait_cycles,
        total_flits=net.stats.flits_sent,
        ordering_violations=0,
        packets=net.stats.packets_sent,
    )


def run_color_scheme(
    rows: int,
    cols: int,
    traffic: list[dict],
    *,
    msg_bytes: int = 128,
    num_colors: int = 24,
    defect: DefectMap | None = None,
) -> SimCaseResult:
    plan = build_ideal_plan(rows, cols, num_colors)
    env = simpy.Environment()
    net = ColorNetwork(
        env=env,
        plan=plan,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        entries_per_color=2,
        flit_bytes=msg_bytes,
        router_pipeline_mode="1_stage",
    )
    if defect is not None:
        apply_defect_map_to_graph(net.graph, defect)
        net.remove_dead_components(defect.dead_cores, defect.dead_links)
        plan, _ = repair_color_plan(plan, net.graph, defect.dead_cores, defect.dead_links)
        net.plan = plan

    net.run_traffic(traffic)
    env.run()
    return SimCaseResult(
        mesh=f"{rows}x{cols}",
        scheme="color_ideal",
        pattern="",
        msg_bytes=msg_bytes,
        makespan_cycles=int(env.now),
        avg_latency=net.stats.avg_latency(),
        avg_link_util=net.avg_link_util(),
        color_buffer_wait_cycles=net.stats.color_buffer_wait_cycles,
        buffer_wait_cycles=net.stats.buffer_wait_cycles,
        link_wait_cycles=net.stats.link_wait_cycles,
        total_flits=net.stats.flits_sent,
        ordering_violations=net.stats.ordering_violations,
        packets=net.stats.packets_sent,
    )


def compare_pattern(
    rows: int,
    cols: int,
    pattern: str,
    *,
    msg_bytes: int = 128,
    num_experts: int = 1,
) -> tuple[SimCaseResult, SimCaseResult]:
    del num_experts
    chunk = msg_bytes
    baseline_traffic = generate_baseline_collective(pattern, rows, cols, chunk)
    ideal_traffic = generate_ideal_collective(pattern, rows, cols, chunk)
    xy = run_xy_baseline(rows, cols, baseline_traffic, msg_bytes=msg_bytes)
    color = run_color_scheme(rows, cols, ideal_traffic, msg_bytes=msg_bytes)
    xy.pattern = pattern
    color.pattern = pattern
    return xy, color


def run_full_study(
    meshes: list[tuple[int, int]],
    output_dir: Path,
    *,
    msg_bytes: int = 128,
) -> list[SimCaseResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SimCaseResult] = []
    for rows, cols in meshes:
        for pattern in PATTERNS:
            xy, color = compare_pattern(rows, cols, pattern, msg_bytes=msg_bytes)
            results.extend([xy, color])
        meta = {
            "mesh": f"{rows}x{cols}",
            "patterns": PATTERNS,
            "msg_bytes": msg_bytes,
        }
        (output_dir / f"mesh_{rows}x{cols}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    return results


def _inject_traffic(env, net: UnifiedNetwork, traffic: list[dict]) -> None:
    from wsesim.network.packet import Packet

    for pkt in traffic:
        delay = int(pkt.get("delay_cycles", 0))
        p = Packet(
            src=int(pkt["src_core"]),
            dst=int(pkt["dst_core"]),
            size_bytes=int(pkt["size_bytes"]),
            payload_type=str(pkt.get("payload", "data")),
        )

        def _send(packet=p, d=delay):
            if d:
                yield env.timeout(d)
            yield env.process(net.send_packet(packet))

        env.process(_send())


def _xy_link_util(net: UnifiedNetwork) -> float:
    if not net.links:
        return 0.0
    total_busy = sum(l.total_busy_cycles for l in net.links.values())
    horizon = max(1, int(net.env.now))
    return total_busy / (len(net.links) * horizon)

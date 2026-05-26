"""Design-space exploration for TDM color subtopology vs ND AllGather makespan."""

from __future__ import annotations

import csv
import html
import itertools
from pathlib import Path

import simpy

from wsesim.network.color_planners import (
    ALL_PLANNERS,
    CONSTRAINT_A,
    CONSTRAINT_AB,
    LINK_UNIVERSE_FULL,
    LINK_UNIVERSE_ND_ACTIVE,
    ColorPlannerConfig,
    build_color_plan,
    enumerate_nd_allgather_routes,
    plan_stats,
    validate_plan,
)
from wsesim.network.collective import generate_collective_traffic
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly

ALGORITHM = "nd_dimension_exchange_allgather"
SLOT_CYCLES = 1
MSG_BYTES = 1024
ILP_TIME_LIMIT_S = 120.0

TOPOLOGY_SPECS: tuple[tuple[str, int, int, int, int], ...] = (
    ("tdm_fb_k4_n2", 4, 4, 4, 2),
    ("tdm_fb_k2_n4", 4, 4, 2, 4),
    ("tdm_fb_k8_n2", 8, 8, 8, 2),
    ("tdm_fb_k4_n3", 8, 8, 4, 3),
    ("tdm_fb_k2_n6", 8, 8, 2, 6),
)

CONSTRAINTS = (CONSTRAINT_A, CONSTRAINT_AB)
LINK_UNIVERSES = (LINK_UNIVERSE_FULL, LINK_UNIVERSE_ND_ACTIVE)


def _build_network(
    rows: int,
    cols: int,
    k: int,
    n: int,
    config: ColorPlannerConfig,
) -> UnifiedNetwork:
    env = simpy.Environment()
    topology = TDMFlatButterfly(
        k=k,
        n=n,
        rows=rows,
        cols=cols,
        color_planner_config=config,
    )
    routing = TDMFlatButterflyRouting(topology=topology)
    return UnifiedNetwork(
        env=env,
        topology=topology,
        routing=routing,
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=rows * cols,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
        slot_cycles=SLOT_CYCLES,
    )


def _run_sim(
    rows: int,
    cols: int,
    k: int,
    n: int,
    config: ColorPlannerConfig,
) -> dict[str, int | float | str | bool]:
    topo = TDMFlatButterfly(k=k, n=n, rows=rows, cols=cols)
    plan, stats = build_color_plan(topo, config)
    routes = enumerate_nd_allgather_routes(topo, topology_hint={"k": k, "n": n})
    validation = validate_plan(plan, topo, routes)

    net = _build_network(rows, cols, k, n, config)
    env = net.env
    traffic = generate_collective_traffic(
        algorithm=ALGORITHM,
        participating_nodes_global=list(range(rows * cols)),
        cores_per_reticle=rows * cols,
        payload_bytes_per_expert=MSG_BYTES,
        num_experts=1,
        topology_hint={"k": k, "n": n},
    )
    for item in traffic:
        delay = int(item.get("delay_cycles", 0))

        def _inject(sim_env: simpy.Environment, delay_cycles: int, pkt: dict) -> simpy.events.Process:
            if delay_cycles > 0:
                yield sim_env.timeout(delay_cycles)
            yield sim_env.process(
                net.send_packet(
                    Packet(
                        src=int(pkt["src_core"]),
                        dst=int(pkt["dst_core"]),
                        size_bytes=int(pkt["size_bytes"]),
                        payload_type=str(pkt["payload"]),
                    )
                )
            )

        env.process(_inject(env, delay, item))

    env.run()
    return {
        "C": plan.C,
        "color_lower_bound": plan.color_lower_bound,
        "balance_ratio": stats.balance_ratio,
        "min_links_per_color": stats.min_links_per_color,
        "max_links_per_color": stats.max_links_per_color,
        "planner_used": stats.planner_used,
        "ilp_fallback_used": stats.ilp_fallback_used,
        "monochrome_rate": validation.monochrome_rate,
        "edge_conflicts": len(validation.edge_conflicts),
        "makespan_cycles": int(env.now),
        "color_buffer_wait_cycles": int(net.stats.color_buffer_wait_cycles),
        "link_wait_cycles": int(net.stats.link_wait_cycles),
        "avg_latency": float(net.stats.avg_latency()),
        "total_flits": int(net.stats.flits_sent),
    }


def _config_id(constraint: str, link_universe: str, planner: str) -> str:
    return f"{constraint}|{link_universe}|{planner}"


def run_dse() -> list[dict[str, int | float | str | bool]]:
    rows_out: list[dict[str, int | float | str | bool]] = []
    for topo_key, rows, cols, k, n in TOPOLOGY_SPECS:
        for constraint, link_universe, planner in itertools.product(
            CONSTRAINTS, LINK_UNIVERSES, ALL_PLANNERS
        ):
            config = ColorPlannerConfig(
                constraint=constraint,
                link_universe=link_universe,
                planner=planner,
                time_limit_s=ILP_TIME_LIMIT_S,
                topology_hint={"k": k, "n": n},
            )
            metrics = _run_sim(rows, cols, k, n, config)
            rows_out.append(
                {
                    "topology": topo_key,
                    "rows": rows,
                    "cols": cols,
                    "k": k,
                    "n": n,
                    "constraint": constraint,
                    "link_universe": link_universe,
                    "planner": planner,
                    "config_id": _config_id(constraint, link_universe, planner),
                    **metrics,
                }
            )
    return rows_out


def _baseline_makespan(rows: list[dict], topo_key: str) -> int:
    for row in rows:
        if (
            row["topology"] == topo_key
            and row["constraint"] == CONSTRAINT_A
            and row["link_universe"] == LINK_UNIVERSE_FULL
            and row["planner"] == "greedy_first_fit"
        ):
            return int(row["makespan_cycles"])
    return 0


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_table(rows: list[dict], topo_key: str) -> str:
    subset = [r for r in rows if r["topology"] == topo_key]
    subset.sort(key=lambda r: int(r["makespan_cycles"]))
    baseline = _baseline_makespan(rows, topo_key) or 1
    lines = [
        "<table>",
        "<tr><th>config</th><th>C</th><th>balance</th><th>makespan</th>"
        "<th>vs baseline</th><th>color wait</th><th>mono rate</th><th>planner</th></tr>",
    ]
    for row in subset:
        ms = int(row["makespan_cycles"])
        speedup = baseline / ms if ms else 0.0
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['config_id']))}</td>"
            f"<td>{row['C']}</td>"
            f"<td>{row['balance_ratio']:.2f}</td>"
            f"<td>{ms}</td>"
            f"<td>{speedup:.3f}x</td>"
            f"<td>{row['color_buffer_wait_cycles']}</td>"
            f"<td>{float(row['monochrome_rate']):.3f}</td>"
            f"<td>{html.escape(str(row['planner_used']))}"
            f"{' (fallback)' if row['ilp_fallback_used'] else ''}</td>"
            "</tr>"
        )
    lines.append("</table>")
    return "\n".join(lines)


def write_html(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = min(rows, key=lambda r: int(r["makespan_cycles"]))
    sections = []
    for topo_key, _, _, _, _ in TOPOLOGY_SPECS:
        sections.append(f"<h2>{html.escape(topo_key)}</h2>")
        sections.append(_render_table(rows, topo_key))

    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>Color DSE — ND AllGather makespan</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
th {{ background: #f4f4f4; }}
code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
<h1>Color 子拓扑 DSE — ND AllGather</h1>
<p>算法: <code>{ALGORITHM}</code> · slot_cycles={SLOT_CYCLES} · msg_bytes={MSG_BYTES}</p>
<p>全局最优 makespan: <strong>{best['makespan_cycles']}</strong>
 ({html.escape(str(best['topology']))} / {html.escape(str(best['config_id']))})</p>
<p>设计空间: 2 constraints × 2 link universes × {len(ALL_PLANNERS)} planners = {2 * 2 * len(ALL_PLANNERS)} configs / topology</p>
{''.join(sections)}
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = run_dse()
    csv_path = root / "results" / "color_dse_allgather.csv"
    html_path = root / "docs" / "color_dse_allgather_report.html"
    write_csv(rows, csv_path)
    write_html(rows, html_path)
    print(f"Wrote {csv_path} ({len(rows)} rows)")
    print(f"Wrote {html_path}")
    best = min(rows, key=lambda r: int(r["makespan_cycles"]))
    print(
        f"Best makespan={best['makespan_cycles']} "
        f"topology={best['topology']} config={best['config_id']}"
    )


if __name__ == "__main__":
    main()

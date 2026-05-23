"""Run allgather comparisons: mesh2d vs TDM flattened butterfly on 4x4."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import simpy

from wsesim.network.collective import generate_collective_traffic
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.topology.mesh2d import Mesh2D
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly

NODE_COUNT = 16
ROWS = 4
COLS = 4
TOPOLOGIES = ("mesh2d_4x4_ps", "tdm_fb_k4_n2", "tdm_fb_k2_n4")
ALGORITHMS = ("direct_allgather", "nd_dimension_exchange_allgather")
SLOT_CYCLES = (1, 4)
MESSAGE_SIZES = (1024, 16 * 1024, 256 * 1024)


def _build_network(topology_key: str, slot_cycles: int) -> UnifiedNetwork:
    env = simpy.Environment()
    if topology_key == "mesh2d_4x4_ps":
        topology = Mesh2D(rows=ROWS, cols=COLS)
        routing = DimensionOrderRouting()
    else:
        _, _, k_token, n_token = topology_key.split("_")
        k = int(k_token[1:])
        n = int(n_token[1:])
        topology = TDMFlatButterfly(k=k, n=n, rows=ROWS, cols=COLS)
        routing = TDMFlatButterflyRouting(topology=topology)
    return UnifiedNetwork(
        env=env,
        topology=topology,
        routing=routing,
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=NODE_COUNT,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
        slot_cycles=slot_cycles,
    )


def _topology_hint(topology_key: str) -> dict[str, int]:
    if topology_key == "mesh2d_4x4_ps":
        return {"rows": ROWS, "cols": COLS}
    _, _, k_token, n_token = topology_key.split("_")
    return {"k": int(k_token[1:]), "n": int(n_token[1:])}


def _run_case(topology_key: str, algorithm: str, slot_cycles: int, msg_bytes: int) -> dict[str, int | float | str]:
    net = _build_network(topology_key=topology_key, slot_cycles=slot_cycles)
    env = net.env
    traffic = generate_collective_traffic(
        algorithm=algorithm,
        participating_nodes_global=list(range(NODE_COUNT)),
        cores_per_reticle=NODE_COUNT,
        payload_bytes_per_expert=msg_bytes,
        num_experts=1,
        topology_hint=_topology_hint(topology_key),
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
    sim_time = max(1.0, float(env.now))
    avg_link_util = (
        sum(link.total_busy_cycles for link in net.links.values()) / (len(net.links) * sim_time)
        if net.links
        else 0.0
    )
    return {
        "topology": topology_key,
        "algorithm": algorithm,
        "slot_cycles": int(slot_cycles),
        "msg_bytes": int(msg_bytes),
        "makespan_cycles": int(env.now),
        "avg_latency": float(net.stats.avg_latency()),
        "avg_link_util": float(avg_link_util),
        "link_wait_cycles": int(net.stats.link_wait_cycles),
        "color_buffer_wait_cycles": int(net.stats.color_buffer_wait_cycles),
        "router_pipeline_cycles": int(net.stats.pipeline_cycles),
        "total_flits": int(net.stats.flits_sent),
    }


def _render_report(rows: list[dict[str, int | float | str]], out_html: Path) -> None:
    best = min(rows, key=lambda r: int(r["makespan_cycles"]))
    worst = max(rows, key=lambda r: int(r["makespan_cycles"]))
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["topology"]), int(row["slot_cycles"]))].append(int(row["makespan_cycles"]))

    avg_makespan = {
        key: sum(values) / max(1, len(values))
        for key, values in grouped.items()
    }
    mesh_avg = avg_makespan.get(("mesh2d_4x4_ps", 1), 0.0)
    tdm_avg = [
        v for (topo, _), v in avg_makespan.items() if topo != "mesh2d_4x4_ps"
    ]
    tdm_mean = sum(tdm_avg) / max(1, len(tdm_avg))
    speedup = (mesh_avg / tdm_mean) if tdm_mean > 0 else 0.0

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['topology']}</td>"
            f"<td>{row['algorithm']}</td>"
            f"<td>{row['slot_cycles']}</td>"
            f"<td>{row['msg_bytes']}</td>"
            f"<td>{row['makespan_cycles']}</td>"
            f"<td>{float(row['avg_latency']):.2f}</td>"
            f"<td>{float(row['avg_link_util']):.3f}</td>"
            f"<td>{row['link_wait_cycles']}</td>"
            f"<td>{row['color_buffer_wait_cycles']}</td>"
            f"<td>{row['router_pipeline_cycles']}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AllGather 4x4 Mesh vs TDM FB</title>
<style>
  :root {{ --bg:#0f1419; --panel:#1a2332; --text:#e8edf4; --muted:#94a3b8; --border:#2d3a4f; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); }}
  header {{ padding:2rem; background:linear-gradient(135deg,#1e293b,#0f172a); border-bottom:1px solid var(--border); }}
  main {{ max-width:1200px; margin:0 auto; padding:1.5rem; }}
  .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:1rem; margin:1rem 0 1.5rem; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:1rem; }}
  .stat .val {{ font-size:1.5rem; font-weight:700; }}
  .stat .lbl {{ color:var(--muted); font-size:.85rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
  th,td {{ border:1px solid var(--border); padding:.55rem .7rem; text-align:left; }}
  th {{ background:#243044; cursor:pointer; }}
  .hint {{ color:var(--muted); font-size:.9rem; margin:0.5rem 0 1rem; }}
</style>
</head>
<body>
<header>
  <h1>4x4 AllGather: Mesh2D vs TDM Flat Butterfly</h1>
  <p>拓扑: mesh2d_4x4_ps / tdm_fb_k4_n2 / tdm_fb_k2_n4; 算法: direct_allgather + nd_dimension_exchange_allgather; slot_cycles: 1,4。</p>
</header>
<main>
  <div class="summary-grid">
    <div class="stat"><div class="val">{len(rows)}</div><div class="lbl">总实验点</div></div>
    <div class="stat"><div class="val">{best['makespan_cycles']}</div><div class="lbl">最优 makespan ({best['topology']}, {best['algorithm']})</div></div>
    <div class="stat"><div class="val">{worst['makespan_cycles']}</div><div class="lbl">最慢 makespan ({worst['topology']}, {worst['algorithm']})</div></div>
    <div class="stat"><div class="val">{speedup:.2f}x</div><div class="lbl">Mesh 平均 / TDM 平均</div></div>
  </div>
  <p class="hint">点击表头可排序。对照拓扑着色文档: <a href="./tdm_fb_4x4_color_subtopology.html">k4_n2</a> / <a href="./tdm_fb_k2_n4_color_subtopology.html">k2_n4</a></p>
  <table id="result-table">
    <thead>
      <tr>
        <th>topology</th><th>algorithm</th><th>slot_cycles</th><th>msg_bytes</th>
        <th>makespan_cycles</th><th>avg_latency</th><th>avg_link_util</th>
        <th>link_wait_cycles</th><th>color_buffer_wait_cycles</th><th>router_pipeline_cycles</th>
      </tr>
    </thead>
    <tbody>
      {"".join(table_rows)}
    </tbody>
  </table>
</main>
<script>
const table = document.getElementById("result-table");
const headers = table.querySelectorAll("th");
headers.forEach((header, idx) => {{
  header.addEventListener("click", () => {{
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const asc = header.dataset.asc !== "1";
    rows.sort((a, b) => {{
      const va = a.children[idx].textContent.trim();
      const vb = b.children[idx].textContent.trim();
      const na = Number(va);
      const nb = Number(vb);
      if (!Number.isNaN(na) && !Number.isNaN(nb)) return asc ? na - nb : nb - na;
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach((r) => tbody.appendChild(r));
    headers.forEach((h) => delete h.dataset.asc);
    header.dataset.asc = asc ? "1" : "0";
  }});
}});
</script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    rows: list[dict[str, int | float | str]] = []
    for topology in TOPOLOGIES:
        slot_values = (1,) if topology == "mesh2d_4x4_ps" else SLOT_CYCLES
        for algorithm in ALGORITHMS:
            for slot_cycles in slot_values:
                for msg_bytes in MESSAGE_SIZES:
                    rows.append(_run_case(topology, algorithm, slot_cycles, msg_bytes))

    out_dir = Path("outputs/allgather_tdm_vs_mesh_4x4")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "topology",
                "algorithm",
                "slot_cycles",
                "msg_bytes",
                "makespan_cycles",
                "avg_latency",
                "avg_link_util",
                "link_wait_cycles",
                "color_buffer_wait_cycles",
                "router_pipeline_cycles",
                "total_flits",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report_path = Path("docs/allgather_tdm_vs_mesh_4x4_report.html")
    _render_report(rows, report_path)
    print(f"wrote {out_csv}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

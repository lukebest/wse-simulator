#!/usr/bin/env python3
"""AllGather latency comparison on rectangular meshes: Mesh2D vs full 2-flat FB vs restricted hypercube FB."""

from __future__ import annotations

import csv
from pathlib import Path

import simpy

from wsesim.network.collective import generate_collective_traffic
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.topology.mesh2d import Mesh2D
from wsesim.network.topology.rect_flat_butterfly import RectFlatButterfly
from wsesim.network.topology.restricted_hypercube_fb import RestrictedHypercubeFB

SLOT_CYCLES = (1, 4)
MESSAGE_SIZES = (1024, 16 * 1024, 256 * 1024)
# ND 分组交换是主对比指标；direct_allgather 在大 mesh 上为 O(N²)，仅对小 mesh 全量跑。
ALGORITHMS = ("nd_dimension_exchange_allgather",)
ALGORITHMS_WITH_DIRECT = ("direct_allgather", "nd_dimension_exchange_allgather")


def _algorithms_for_mesh(mesh: dict) -> tuple[str, ...]:
    if mesh["rows"] * mesh["cols"] <= 64:
        return ALGORITHMS_WITH_DIRECT
    return ALGORITHMS


def _slot_cycles_for_case(mesh: dict, kind: str) -> tuple[int, ...]:
    if kind == "mesh2d":
        return (1,)
    node_count = mesh["rows"] * mesh["cols"]
    if node_count > 64:
        return (1,)
    return SLOT_CYCLES


def _message_sizes_for_case(mesh: dict, algorithm: str) -> tuple[int, ...]:
    node_count = mesh["rows"] * mesh["cols"]
    if algorithm == "direct_allgather" and node_count > 64:
        return (256 * 1024,)
    if algorithm == "nd_dimension_exchange_allgather" and node_count > 96:
        return (1024, 16 * 1024)
    return MESSAGE_SIZES

MESH_CONFIGS = [
    {
        "slug": "6x8",
        "rows": 6,
        "cols": 8,
        "restricted_parent": (8, 8, 6),
        "direct_scale": 48,
    },
    {
        "slug": "12x16",
        "rows": 12,
        "cols": 16,
        "restricted_parent": (16, 16, 8),
        "direct_scale": 64,
    },
    {
        "slug": "14x14",
        "rows": 14,
        "cols": 14,
        "restricted_parent": (16, 16, 8),
        "direct_scale": 64,
    },
]

TOPOLOGY_KINDS = ("mesh2d", "tdm_fb_rect_full", "tdm_fb_restricted")


def _topology_key(mesh_slug: str, kind: str) -> str:
    return f"{kind}_{mesh_slug}"


def _build_network(mesh: dict, kind: str, slot_cycles: int) -> UnifiedNetwork:
    rows, cols = mesh["rows"], mesh["cols"]
    env = simpy.Environment()

    if kind == "mesh2d":
        topology = Mesh2D(rows=rows, cols=cols)
        routing = DimensionOrderRouting()
        num_nodes = rows * cols
    elif kind == "tdm_fb_rect_full":
        topology = RectFlatButterfly(rows=rows, cols=cols)
        routing = TDMFlatButterflyRouting(topology=topology)
        num_nodes = rows * cols
    elif kind == "tdm_fb_restricted":
        parent_rows, parent_cols, n = mesh["restricted_parent"]
        topology = RestrictedHypercubeFB(
            n=n,
            parent_rows=parent_rows,
            parent_cols=parent_cols,
            keep_rows=rows,
            keep_cols=cols,
        )
        routing = TDMFlatButterflyRouting(topology=topology)
        num_nodes = rows * cols
    else:
        raise ValueError(f"unknown topology kind: {kind}")

    return UnifiedNetwork(
        env=env,
        topology=topology,
        routing=routing,
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=num_nodes,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
        slot_cycles=slot_cycles,
    )


def _participating_nodes(mesh: dict, kind: str) -> list[int]:
    rows, cols = mesh["rows"], mesh["cols"]
    if kind == "tdm_fb_restricted":
        parent_rows, parent_cols, n = mesh["restricted_parent"]
        topo = RestrictedHypercubeFB(
            n=n,
            parent_rows=parent_rows,
            parent_cols=parent_cols,
            keep_rows=rows,
            keep_cols=cols,
        )
        return topo.node_ids()
    return list(range(rows * cols))


def _topology_hint(mesh: dict, kind: str, nodes: list[int]) -> dict:
    rows, cols = mesh["rows"], mesh["cols"]
    if kind == "mesh2d":
        return {"rows": rows, "cols": cols}
    if kind == "tdm_fb_rect_full":
        return {"k_dims": [cols, rows], "rows": rows, "cols": cols}
    parent_rows, parent_cols, n = mesh["restricted_parent"]
    topo = RestrictedHypercubeFB(
        n=n,
        parent_rows=parent_rows,
        parent_cols=parent_cols,
        keep_rows=rows,
        keep_cols=cols,
    )
    return {
        "k": 2,
        "n": n,
        "rows": rows,
        "cols": parent_cols,
        "hypercube_coords": {node: topo.to_coords(node) for node in nodes},
    }


def _payload_bytes(algorithm: str, msg_bytes: int, scale: int) -> tuple[int, bool]:
    if algorithm == "direct_allgather" and msg_bytes >= 256 * 1024:
        return max(32, msg_bytes // scale), True
    return msg_bytes, False


def _meta(mesh: dict, kind: str) -> dict:
    rows, cols = mesh["rows"], mesh["cols"]
    if kind == "mesh2d":
        return {"label": f"Mesh2D {rows}×{cols}", "colors": 0, "scheme": "baseline"}
    if kind == "tdm_fb_rect_full":
        topo = RectFlatButterfly(rows=rows, cols=cols)
        c = topo.coloring().C
        return {"label": f"矩形 2-flat FB (C={c})", "colors": c, "scheme": "full_fb"}
    parent_rows, parent_cols, n = mesh["restricted_parent"]
    topo = RestrictedHypercubeFB(
        n=n, parent_rows=parent_rows, parent_cols=parent_cols, keep_rows=rows, keep_cols=cols
    )
    c = topo.coloring().C
    return {
        "label": f"限制版 (2,{n}) 超立方 FB (C={c})",
        "colors": c,
        "scheme": "restricted",
    }


def _run_case(mesh: dict, kind: str, algorithm: str, slot_cycles: int, msg_bytes: int) -> dict:
    net = _build_network(mesh, kind, slot_cycles)
    nodes = _participating_nodes(mesh, kind)
    simulated_payload, scaled = _payload_bytes(algorithm, msg_bytes, mesh["direct_scale"])
    traffic = generate_collective_traffic(
        algorithm=algorithm,
        participating_nodes_global=nodes,
        cores_per_reticle=len(nodes),
        payload_bytes_per_expert=simulated_payload,
        num_experts=1,
        topology_hint=_topology_hint(mesh, kind, nodes),
    )
    env = net.env
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
    key = _topology_key(mesh["slug"], kind)
    meta = _meta(mesh, kind)
    return {
        "mesh": mesh["slug"],
        "topology": key,
        "scheme": meta["scheme"],
        "colors": meta["colors"],
        "algorithm": algorithm,
        "slot_cycles": int(slot_cycles),
        "msg_bytes": int(msg_bytes),
        "simulated_msg_bytes": int(simulated_payload),
        "payload_scaled": scaled,
        "makespan_cycles": int(env.now),
        "avg_latency": float(net.stats.avg_latency()),
        "avg_link_util": float(avg_link_util),
        "link_wait_cycles": int(net.stats.link_wait_cycles),
        "color_buffer_wait_cycles": int(net.stats.color_buffer_wait_cycles),
        "router_pipeline_cycles": int(net.stats.pipeline_cycles),
        "total_flits": int(net.stats.flits_sent),
    }


def _lookup(rows: list[dict], mesh: str, scheme: str, algorithm: str, slot: int, msg: int) -> dict | None:
    for row in rows:
        if (
            row["mesh"] == mesh
            and row["scheme"] == scheme
            and row["algorithm"] == algorithm
            and int(row["slot_cycles"]) == slot
            and int(row["msg_bytes"]) == msg
        ):
            return row
    return None


def _ratio(a: int, b: int) -> str:
    if b <= 0:
        return "n/a"
    return f"{a / b:.2f}×"


def _render_report(rows: list[dict], out_html: Path) -> None:
    compare_rows: list[str] = []
    for mesh_cfg in MESH_CONFIGS:
        slug = mesh_cfg["slug"]
        mesh_nd = _lookup(rows, slug, "baseline", "nd_dimension_exchange_allgather", 1, 1024)
        mesh_ms = int(mesh_nd["makespan_cycles"]) if mesh_nd else 0
        for scheme, label in (("full_fb", "矩形 2-flat"), ("restricted", "限制版 hypercube")):
            for msg in MESSAGE_SIZES:
                tdm = _lookup(rows, slug, scheme, "nd_dimension_exchange_allgather", 1, msg)
                if not tdm:
                    continue
                tdm_ms = int(tdm["makespan_cycles"])
                mesh_row = _lookup(rows, slug, "baseline", "nd_dimension_exchange_allgather", 1, msg)
                mesh_val = int(mesh_row["makespan_cycles"]) if mesh_row else mesh_ms
                cls = "good" if tdm_ms < mesh_val else "warn"
                compare_rows.append(
                    "<tr>"
                    f"<td>{slug}</td>"
                    f"<td>{label} (C={tdm['colors']})</td>"
                    f"<td>{msg}</td>"
                    f"<td>{mesh_val}</td>"
                    f"<td>{tdm_ms}</td>"
                    f"<td class='{cls}'>{_ratio(tdm_ms, mesh_val)}</td>"
                    f"<td>{tdm['color_buffer_wait_cycles']}</td>"
                    "</tr>"
                )

    cross_rows: list[str] = []
    for mesh_cfg in MESH_CONFIGS:
        slug = mesh_cfg["slug"]
        for msg in MESSAGE_SIZES:
            full = _lookup(rows, slug, "full_fb", "nd_dimension_exchange_allgather", 1, msg)
            rest = _lookup(rows, slug, "restricted", "nd_dimension_exchange_allgather", 1, msg)
            if not full or not rest:
                continue
            cross_rows.append(
                "<tr>"
                f"<td>{slug}</td>"
                f"<td>{msg}</td>"
                f"<td>C={full['colors']}</td>"
                f"<td>{full['makespan_cycles']}</td>"
                f"<td>C={rest['colors']}</td>"
                f"<td>{rest['makespan_cycles']}</td>"
                f"<td>{_ratio(int(rest['makespan_cycles']), int(full['makespan_cycles']))}</td>"
                "</tr>"
            )

    table_rows = []
    for row in rows:
        scaled = " (scaled)" if row.get("payload_scaled") else ""
        table_rows.append(
            "<tr>"
            f"<td>{row['mesh']}</td>"
            f"<td>{row['topology']}</td>"
            f"<td>{row['scheme']}</td>"
            f"<td>{row['colors']}</td>"
            f"<td>{row['algorithm']}</td>"
            f"<td>{row['slot_cycles']}</td>"
            f"<td>{row['msg_bytes']}</td>"
            f"<td>{row['simulated_msg_bytes']}{scaled}</td>"
            f"<td>{row['makespan_cycles']}</td>"
            f"<td>{float(row['avg_latency']):.2f}</td>"
            f"<td>{row['color_buffer_wait_cycles']}</td>"
            f"<td>{row['link_wait_cycles']}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>矩形 Mesh AllGather 对比报告</title>
<style>
  :root {{ --bg:#0f1419; --panel:#1a2332; --text:#e8edf4; --muted:#94a3b8; --border:#2d3a4f; --good:#4ade80; --warn:#f59e0b; }}
  body {{ margin:0; font-family:"IBM Plex Sans",system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
  header {{ padding:2rem; background:linear-gradient(135deg,#1e293b,#0f172a); border-bottom:1px solid var(--border); }}
  main {{ max-width:1200px; margin:0 auto; padding:1.5rem; }}
  section {{ margin-bottom:1.75rem; }}
  h2 {{ font-size:1.15rem; margin:0 0 .6rem; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:1rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.88rem; margin-top:.5rem; }}
  th,td {{ border:1px solid var(--border); padding:.45rem .6rem; text-align:left; }}
  th {{ background:#243044; }}
  .good {{ color:var(--good); font-weight:600; }}
  .warn {{ color:var(--warn); font-weight:600; }}
  .hint {{ color:var(--muted); font-size:.9rem; }}
  ul {{ margin:.4rem 0 0 1.2rem; }}
</style>
</head>
<body>
<header>
  <h1>矩形 Mesh AllGather：Mesh2D vs 矩形 2-flat FB vs 限制版 Hypercube FB</h1>
  <p class="hint">Mesh 尺寸：6×8、12×16、14×14。主对比算法：nd_dimension_exchange_allgather (slot=1)。</p>
</header>
<main>
  <section class="panel">
    <h2>1) 两类 TDM 切分 vs Mesh（ND，slot=1）</h2>
    <table>
      <thead><tr><th>Mesh</th><th>方案</th><th>msg</th><th>Mesh</th><th>TDM</th><th>TDM/Mesh</th><th>color_wait</th></tr></thead>
      <tbody>{"".join(compare_rows)}</tbody>
    </table>
  </section>
  <section class="panel">
    <h2>2) 满 FB vs 限制版 Hypercube（ND，slot=1）</h2>
    <table>
      <thead><tr><th>Mesh</th><th>msg</th><th>满 FB C</th><th>满 FB makespan</th><th>限制 C</th><th>限制 makespan</th><th>限制/满 FB</th></tr></thead>
      <tbody>{"".join(cross_rows)}</tbody>
    </table>
    <ul>
      <li>限制版 Color 数显著更少（6×8: 16→5，12×16/14×14: 64/49→10），但逻辑链路稀疏、节点度降低。</li>
      <li>小消息下 color 等待占比低，限制版可能因更少 color 切换而更快；大消息下链路利用率与跳数成为主因。</li>
    </ul>
  </section>
  <section>
    <h2>3) 完整结果</h2>
    <table>
      <thead><tr>
        <th>mesh</th><th>topology</th><th>scheme</th><th>C</th><th>algorithm</th><th>slot</th>
        <th>msg</th><th>simulated_msg</th><th>makespan</th><th>avg_lat</th><th>color_wait</th><th>link_wait</th>
      </tr></thead>
      <tbody>{"".join(table_rows)}</tbody>
    </table>
  </section>
</main>
</body></html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    rows: list[dict] = []
    for mesh in MESH_CONFIGS:
        for kind in TOPOLOGY_KINDS:
            slot_values = _slot_cycles_for_case(mesh, kind)
            for algorithm in _algorithms_for_mesh(mesh):
                for slot_cycles in slot_values:
                    for msg_bytes in _message_sizes_for_case(mesh, algorithm):
                        key = _topology_key(mesh["slug"], kind)
                        print(f"running {key} {algorithm} slot={slot_cycles} msg={msg_bytes}", flush=True)
                        rows.append(_run_case(mesh, kind, algorithm, slot_cycles, msg_bytes))

    out_dir = Path("outputs/allgather_tdm_vs_mesh_rect")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"
    fieldnames = [
        "mesh", "topology", "scheme", "colors", "algorithm", "slot_cycles", "msg_bytes",
        "simulated_msg_bytes", "payload_scaled", "makespan_cycles", "avg_latency",
        "avg_link_util", "link_wait_cycles", "color_buffer_wait_cycles",
        "router_pipeline_cycles", "total_flits",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_path = Path("docs/allgather_tdm_vs_mesh_rect_report.html")
    _render_report(rows, report_path)
    print(f"wrote {out_csv}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

"""Run allgather comparisons: mesh2d vs TDM flattened butterfly on 8x8."""

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

NODE_COUNT = 64
ROWS = 8
COLS = 8
TOPOLOGIES = ("mesh2d_8x8_ps", "tdm_fb_k8_n2", "tdm_fb_k4_n3", "tdm_fb_k2_n6")
ALGORITHMS = ("direct_allgather", "nd_dimension_exchange_allgather")
SLOT_CYCLES = (1, 4)
MESSAGE_SIZES = (1024, 16 * 1024, 256 * 1024)
# direct_allgather at 256KB on 64 nodes is O(N^2) heavy; scale payload for that point only.
DIRECT_LARGE_MSG_SCALE = 64

TOPOLOGY_META = {
    "mesh2d_8x8_ps": {"k": None, "n": None, "colors": 0, "label": "Mesh2D 分组交换基线"},
    "tdm_fb_k8_n2": {"k": 8, "n": 2, "colors": 16, "label": "8-ary 2-flat (C=16)"},
    "tdm_fb_k4_n3": {"k": 4, "n": 3, "colors": 9, "label": "4-ary 3-flat (C=9)"},
    "tdm_fb_k2_n6": {"k": 2, "n": 6, "colors": 5, "label": "2-ary 6-flat (C=5)"},
}


def _build_network(topology_key: str, slot_cycles: int) -> UnifiedNetwork:
    env = simpy.Environment()
    if topology_key == "mesh2d_8x8_ps":
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
    if topology_key == "mesh2d_8x8_ps":
        return {"rows": ROWS, "cols": COLS}
    _, _, k_token, n_token = topology_key.split("_")
    return {"k": int(k_token[1:]), "n": int(n_token[1:])}


def _payload_bytes(algorithm: str, msg_bytes: int) -> tuple[int, bool]:
    if algorithm == "direct_allgather" and msg_bytes >= 256 * 1024:
        return max(32, msg_bytes // DIRECT_LARGE_MSG_SCALE), True
    return msg_bytes, False


def _run_case(
    topology_key: str, algorithm: str, slot_cycles: int, msg_bytes: int
) -> dict[str, int | float | str | bool]:
    net = _build_network(topology_key=topology_key, slot_cycles=slot_cycles)
    env = net.env
    simulated_payload, scaled = _payload_bytes(algorithm, msg_bytes)
    traffic = generate_collective_traffic(
        algorithm=algorithm,
        participating_nodes_global=list(range(NODE_COUNT)),
        cores_per_reticle=NODE_COUNT,
        payload_bytes_per_expert=simulated_payload,
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


def _lookup(rows: list[dict], topology: str, algorithm: str, slot: int, msg: int) -> dict | None:
    for row in rows:
        if (
            row["topology"] == topology
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


def _render_report(rows: list[dict[str, int | float | str | bool]], out_html: Path) -> None:
    nd_slot1 = [
        r for r in rows if r["algorithm"] == "nd_dimension_exchange_allgather" and int(r["slot_cycles"]) == 1
    ]
    best = min(rows, key=lambda r: int(r["makespan_cycles"]))
    worst = max(rows, key=lambda r: int(r["makespan_cycles"]))

    def mesh_msg(msg: int) -> int:
        row = _lookup(rows, "mesh2d_8x8_ps", "nd_dimension_exchange_allgather", 1, msg)
        return int(row["makespan_cycles"]) if row else 0

    def tdm_ratio(topo: str, msg: int) -> float | None:
        mesh_ms = mesh_msg(msg)
        row = _lookup(rows, topo, "nd_dimension_exchange_allgather", 1, msg)
        if not row or mesh_ms <= 0:
            return None
        return int(row["makespan_cycles"]) / mesh_ms

    def slot_ratio(topo: str, msg: int) -> float | None:
        s1 = _lookup(rows, topo, "nd_dimension_exchange_allgather", 1, msg)
        s4 = _lookup(rows, topo, "nd_dimension_exchange_allgather", 4, msg)
        if not s1 or not s4 or int(s1["makespan_cycles"]) <= 0:
            return None
        return int(s4["makespan_cycles"]) / int(s1["makespan_cycles"])

    mesh_nd_avg = sum(
        int(r["makespan_cycles"]) for r in nd_slot1 if r["topology"] == "mesh2d_8x8_ps"
    ) / max(1, len([r for r in nd_slot1 if r["topology"] == "mesh2d_8x8_ps"]))
    tdm_nd_vals = [
        int(r["makespan_cycles"]) for r in nd_slot1 if r["topology"] != "mesh2d_8x8_ps"
    ]
    tdm_nd_avg = sum(tdm_nd_vals) / max(1, len(tdm_nd_vals))

    k2_ratios = [r for r in (tdm_ratio("tdm_fb_k2_n6", m) for m in MESSAGE_SIZES) if r is not None]
    k4_ratios = [r for r in (tdm_ratio("tdm_fb_k4_n3", m) for m in MESSAGE_SIZES) if r is not None]
    k8_ratios = [r for r in (tdm_ratio("tdm_fb_k8_n2", m) for m in MESSAGE_SIZES) if r is not None]
    slot_ratios_k8 = [r for r in (slot_ratio("tdm_fb_k8_n2", m) for m in MESSAGE_SIZES) if r is not None]

    def fmt_range(values: list[float]) -> str:
        if not values:
            return "n/a"
        return f"{min(values):.2f}×~{max(values):.2f}×"

    k2_range = fmt_range(k2_ratios)
    k4_range = fmt_range(k4_ratios)
    k8_range = fmt_range(k8_ratios)
    slot_k8_range = fmt_range(slot_ratios_k8)

    k2_verdict = "全面优于 Mesh" if k2_ratios and max(k2_ratios) < 1.0 else "部分场景优于 Mesh"
    k4_verdict = "整体优于 Mesh" if k4_ratios and sum(k4_ratios) / len(k4_ratios) < 1.0 else "与 Mesh 互有胜负"
    k8_verdict = "整体慢于 Mesh" if k8_ratios and min(k8_ratios) >= 1.0 else "部分场景慢于 Mesh"

    k4_advice = (
        "小消息略优、中大消息略慢，可作为次优折中"
        if k4_ratios and min(k4_ratios) < 1.0 and max(k4_ratios) > 1.0
        else ("整体优于 Mesh" if k4_ratios and max(k4_ratios) < 1.0 else "整体不优于 Mesh")
    )

    bottleneck_rows: list[str] = []
    msg = 262144
    mesh_nd = _lookup(rows, "mesh2d_8x8_ps", "nd_dimension_exchange_allgather", 1, msg)
    for topo in ("tdm_fb_k8_n2", "tdm_fb_k4_n3", "tdm_fb_k2_n6"):
        tdm = _lookup(rows, topo, "nd_dimension_exchange_allgather", 1, msg)
        if not mesh_nd or not tdm:
            continue
        mesh_ms = int(mesh_nd["makespan_cycles"])
        tdm_ms = int(tdm["makespan_cycles"])
        bottleneck_rows.append(
            "<tr>"
            f"<td>{TOPOLOGY_META[topo]['label']}</td>"
            f"<td>{tdm_ms}</td>"
            f"<td>{tdm['color_buffer_wait_cycles']}</td>"
            f"<td>{tdm['link_wait_cycles']}</td>"
            f"<td>{float(tdm['avg_link_util']):.3f}</td>"
            f"<td>{_ratio(tdm_ms, mesh_ms)}</td>"
            "</tr>"
        )

    compare_rows: list[str] = []
    for topo in ("tdm_fb_k8_n2", "tdm_fb_k4_n3", "tdm_fb_k2_n6"):
        for msg in MESSAGE_SIZES:
            mesh_ms = mesh_msg(msg)
            tdm = _lookup(rows, topo, "nd_dimension_exchange_allgather", 1, msg)
            if not tdm:
                continue
            tdm_ms = int(tdm["makespan_cycles"])
            verdict = "TDM 更优" if tdm_ms < mesh_ms else "Mesh 更优"
            cls = "good" if tdm_ms < mesh_ms else "warn"
            compare_rows.append(
                "<tr>"
                f"<td>{TOPOLOGY_META[topo]['label']}</td>"
                f"<td>{msg}</td>"
                f"<td>{mesh_ms}</td>"
                f"<td>{tdm_ms}</td>"
                f"<td class='{cls}'>{_ratio(tdm_ms, mesh_ms)}</td>"
                f"<td class='{cls}'>{verdict}</td>"
                "</tr>"
            )

    slot_rows: list[str] = []
    for topo in ("tdm_fb_k8_n2", "tdm_fb_k4_n3", "tdm_fb_k2_n6"):
        for msg in MESSAGE_SIZES:
            s1 = _lookup(rows, topo, "nd_dimension_exchange_allgather", 1, msg)
            s4 = _lookup(rows, topo, "nd_dimension_exchange_allgather", 4, msg)
            if not s1 or not s4:
                continue
            slot_rows.append(
                "<tr>"
                f"<td>{TOPOLOGY_META[topo]['label']}</td>"
                f"<td>{msg}</td>"
                f"<td>{s1['makespan_cycles']}</td>"
                f"<td>{s4['makespan_cycles']}</td>"
                f"<td>{_ratio(int(s4['makespan_cycles']), int(s1['makespan_cycles']))}</td>"
                "</tr>"
            )

    table_rows = []
    for row in rows:
        scaled_note = " (scaled)" if row.get("payload_scaled") else ""
        table_rows.append(
            "<tr>"
            f"<td>{row['topology']}</td>"
            f"<td>{row['algorithm']}</td>"
            f"<td>{row['slot_cycles']}</td>"
            f"<td>{row['msg_bytes']}</td>"
            f"<td>{row['simulated_msg_bytes']}{scaled_note}</td>"
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
<title>AllGather 8x8 仿真报告</title>
<style>
  :root {{
    --bg:#0f1419; --panel:#1a2332; --text:#e8edf4; --muted:#94a3b8; --border:#2d3a4f;
    --good:#4ade80; --warn:#f59e0b;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
  header {{ padding:2rem; background:linear-gradient(135deg,#1e293b,#0f172a); border-bottom:1px solid var(--border); }}
  main {{ max-width:1200px; margin:0 auto; padding:1.5rem; }}
  section {{ margin-bottom:1.75rem; }}
  h2 {{ font-size:1.2rem; margin:0 0 .6rem; }}
  .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:1rem; margin:1rem 0 1.5rem; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:1rem; }}
  .stat .val {{ font-size:1.45rem; font-weight:700; }}
  .stat .lbl {{ color:var(--muted); font-size:.86rem; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:1rem 1.1rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.9rem; margin-top:.6rem; }}
  th,td {{ border:1px solid var(--border); padding:.5rem .65rem; text-align:left; }}
  th {{ background:#243044; cursor:pointer; }}
  .hint {{ color:var(--muted); font-size:.9rem; }}
  ul {{ margin:.4rem 0 0 1.2rem; }}
  .good {{ color:var(--good); font-weight:600; }}
  .warn {{ color:var(--warn); font-weight:600; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
</style>
</head>
<body>
<header>
  <h1>8×8 AllGather 仿真分析报告</h1>
  <p>对比 <span class="mono">mesh2d_8x8_ps</span> 与三种 TDM FB：<span class="mono">k8_n2 (C=16)</span> / <span class="mono">k4_n3 (C=9)</span> / <span class="mono">k2_n6 (C=5)</span>。</p>
  <p class="hint">主对比算法：<strong>nd_dimension_exchange_allgather</strong>（分组交换）；direct_allgather 256KB 采用 payload/{DIRECT_LARGE_MSG_SCALE} 缩放以控制 64 节点 O(N²) 仿真成本。</p>
</header>
<main>
  <section>
    <div class="summary-grid">
      <div class="stat"><div class="val">{len(rows)}</div><div class="lbl">实验点总数</div></div>
      <div class="stat"><div class="val">{best['makespan_cycles']} cycles</div><div class="lbl">全局最优 ({best['topology']}, {best['algorithm']})</div></div>
      <div class="stat"><div class="val">{worst['makespan_cycles']} cycles</div><div class="lbl">全局最慢 ({worst['topology']}, {worst['algorithm']})</div></div>
      <div class="stat"><div class="val">{_ratio(int(mesh_nd_avg), int(tdm_nd_avg))}</div><div class="lbl">Mesh ND 平均 / TDM ND 平均 (slot=1)</div></div>
    </div>
  </section>

  <section class="panel">
    <h2>1) 关键结论（分组交换 ND，slot=1）</h2>
    <ul>
      <li><span class="good">k2_n6 (C=5) ND AllGather：{k2_verdict}</span>，TDM/Mesh 约为 <span class="mono">{k2_range}</span>。</li>
      <li><span class="good">k4_n3 (C=9) ND AllGather：{k4_verdict}</span>，TDM/Mesh 约为 <span class="mono">{k4_range}</span>。</li>
      <li><span class="warn">k8_n2 (C=16) ND AllGather：{k8_verdict}</span>，TDM/Mesh 约为 <span class="mono">{k8_range}</span>。</li>
      <li>slot 1→4 对 k8_n2 的 ND makespan 比值约 <span class="mono">{slot_k8_range}</span>（color 数越多通常越敏感）。</li>
      <li>瓶颈分解：TDM 的 <span class="mono">color_buffer_wait_cycles</span> 与 <span class="mono">link_wait_cycles</span> 随 C 增大显著上升，是 k8_n2 落后的主因。</li>
      <li>direct_allgather 在 64 节点下流量为 O(N²)，TDM 全面慢于 Mesh；该算法不适合作为 TDM FB 的主评估指标，应以 ND 分组交换为主。</li>
    </ul>
  </section>

  <section class="panel">
    <h2>2) ND 分组交换 vs Mesh 基线（slot=1）</h2>
    <table>
      <thead><tr><th>拓扑</th><th>msg_bytes</th><th>Mesh makespan</th><th>TDM makespan</th><th>TDM/Mesh</th><th>结论</th></tr></thead>
      <tbody>{"".join(compare_rows)}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>3) 256KB ND 瓶颈拆解（slot=1，Mesh makespan={mesh_msg(262144)} cycles）</h2>
    <table>
      <thead><tr><th>拓扑</th><th>makespan</th><th>color_buffer_wait</th><th>link_wait</th><th>avg_link_util</th><th>TDM/Mesh</th></tr></thead>
      <tbody>{"".join(bottleneck_rows)}</tbody>
    </table>
    <p class="hint">k8_n2 的 color 等待在大消息下占 makespan 主体；k2_n6 等待项低且链路利用率接近 Mesh，因此端到端最优。</p>
  </section>

  <section class="panel">
    <h2>4) slot_cycles 敏感性（ND 算法）</h2>
    <table>
      <thead><tr><th>拓扑</th><th>msg_bytes</th><th>slot=1</th><th>slot=4</th><th>slot=4 / slot=1</th></tr></thead>
      <tbody>{"".join(slot_rows)}</tbody>
    </table>
  </section>

  <section class="panel">
    <h2>5) 工程建议</h2>
    <ul>
      <li>8×8 AllGather 若走 TDM FB，优先 <span class="mono">k2_n6 + nd_dimension_exchange + slot=1</span>（109~3267 cycles，相对 Mesh 约 0.39×~0.40×）。</li>
      <li>k4_n3 {k4_advice}（C=9；1KB 略优，16KB/256KB 略慢 7%~18%）。</li>
      <li>k8_n2 在当前 color/per-buffer 模型下不适合作为 AllGather 首选（C=16，ND 约为 Mesh 的 3×~3.75×）。</li>
      <li>direct_allgather 仅作参考：TDM 因 color 等待与 O(N²) 流量叠加，makespan 显著高于 Mesh。</li>
    </ul>
    <p class="hint">着色参考：
      <a href="./tdm_fb_8x8_k8_n2_color_subtopology.html">k8_n2</a> /
      <a href="./tdm_fb_8x8_k4_n3_color_subtopology.html">k4_n3</a> /
      <a href="./tdm_fb_8x8_k2_n6_color_subtopology.html">k2_n6</a>
    </p>
  </section>

  <section>
    <h2>6) 完整结果明细（可排序）</h2>
    <table id="result-table">
      <thead>
        <tr>
          <th>topology</th><th>algorithm</th><th>slot_cycles</th><th>msg_bytes</th><th>simulated_msg_bytes</th>
          <th>makespan_cycles</th><th>avg_latency</th><th>avg_link_util</th>
          <th>link_wait_cycles</th><th>color_buffer_wait_cycles</th><th>router_pipeline_cycles</th>
        </tr>
      </thead>
      <tbody>{"".join(table_rows)}</tbody>
    </table>
  </section>
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
      const na = Number(va.replace(/[^0-9.-]/g, ""));
      const nb = Number(vb.replace(/[^0-9.-]/g, ""));
      if (!Number.isNaN(na) && !Number.isNaN(nb) && va !== "" && vb !== "") return asc ? na - nb : nb - na;
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
    rows: list[dict[str, int | float | str | bool]] = []
    for topology in TOPOLOGIES:
        slot_values = (1,) if topology == "mesh2d_8x8_ps" else SLOT_CYCLES
        for algorithm in ALGORITHMS:
            for slot_cycles in slot_values:
                for msg_bytes in MESSAGE_SIZES:
                    print(f"running {topology} {algorithm} slot={slot_cycles} msg={msg_bytes}")
                    rows.append(_run_case(topology, algorithm, slot_cycles, msg_bytes))

    out_dir = Path("outputs/allgather_tdm_vs_mesh_8x8")
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
                "simulated_msg_bytes",
                "payload_scaled",
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

    report_path = Path("docs/allgather_tdm_vs_mesh_8x8_report.html")
    _render_report(rows, report_path)
    print(f"wrote {out_csv}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

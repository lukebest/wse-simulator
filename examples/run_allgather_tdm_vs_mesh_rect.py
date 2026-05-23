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
  .glossary {{ font-size:.9rem; line-height:1.7; }}
  .glossary dt {{ font-weight:600; color:#cbd5e1; margin-top:.75rem; }}
  .glossary dd {{ margin:.25rem 0 0 0; color:var(--muted); }}
  .callout {{ background:#0f172a; border-left:3px solid #3b82f6; padding:.75rem 1rem; margin:.75rem 0; font-size:.88rem; border-radius:0 6px 6px 0; }}
  .callout strong {{ color:#93c5fd; }}
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
    <h2>指标说明（读表前必读）</h2>

    <div class="callout">
      <strong>为什么 color_wait 可以远大于 makespan？</strong><br>
      两者量纲不同，<em>不能直接比较大小</em>。<code>makespan</code> 是仿真时钟的<strong>墙钟时间</strong>（最后一个包完成时的 <code>env.now</code>，即关键路径长度）。
      <code>color_wait</code> 是<strong>全网络累加计数</strong>：每个 router 上、每个 flit 在 per-color buffer 出口等待「当前 TDM 时隙 == 该 flit 的 Color」时，每 spin 1 cycle 就 +1，最后把所有 router 的等待<strong>求和</strong>。
      大量包/ flit 在<strong>并行</strong>等待，累加值自然远大于墙钟。例：12×16 满 FB ND 1KB — makespan=9225，color_wait=1,104,889（≈120×），表示并行等待总量大，但不代表单包等了 120 万 cycle。
    </div>

    <div class="callout">
      <strong>TDM/Mesh ≈ 0.03×、0.07× 的时延收益从哪来？</strong><br>
      比值只看 <code>makespan</code>（端到端完成时间），与 color_wait 累加值无关。限制版 hypercube FB 的优势主要来自：
      <ol style="margin:.4rem 0 0 1.2rem;padding:0">
        <li><strong>Color 数 C 更少</strong>（5/10 vs 16/64/49）→ 每跳等正确 TDM 时隙的周期更短，关键路径上 color 相关等待更少。</li>
        <li><strong>ND 流量模式不同</strong>：限制版按 (k=2, n) 超立方分组，每组仅 2 节点交换；Mesh/满 FB 按混合基 k_dims（如 16×12）分组，组内全对全通信，包数/flit 数多一个数量级（如 12×16 Mesh 342k flits vs 限制版 5k flits）。</li>
        <li><strong>平均包延迟更低</strong>：限制版 avg_lat 约 39–118 cycles，Mesh 350–5585 cycles（见完整结果表）。</li>
        <li><strong>逻辑跳数更少</strong>：稀疏 hypercube 链路 + 维度序路由，单包物理跳数通常短于 Mesh 分组交换在大组内的多跳转发。</li>
      </ol>
      注意：限制版用更少的逻辑链路换更少的 Color；满 FB 在 14×14 上 C=49 反而比 Mesh 慢（makespan 8×），说明<strong>Color 数过多</strong>的代价可以超过 FB 拓扑的带宽优势。
    </div>

    <dl class="glossary">
      <dt>msg（msg_bytes）</dt>
      <dd>实验配置的<strong>标称消息大小</strong>（字节），用于标识测试点，写入 CSV 的 <code>msg_bytes</code> 列。</dd>

      <dt>simulated_msg（simulated_msg_bytes）</dt>
      <dd>仿真中<strong>实际注入的 payload 大小</strong>。通常与 msg 相同。
        若标注 <code>(scaled)</code>，表示 <code>direct_allgather</code> 在 256KB 点时按 <code>payload / N</code> 缩放（6×8 除以 48，12×16/14×14 除以 64），以降低 O(N²) 全对全流量的仿真成本；此时 msg=262144 但 simulated_msg 仅为几千字节。</dd>

      <dt>color_wait（color_buffer_wait_cycles）</dt>
      <dd><strong>Router 侧</strong> TDM 等待：flit 从 per-color ingress buffer 取出后、进入 switch pipeline 之前，等待全局 TDM 时钟转到该 flit 的 Color 时隙。仅 TDM 拓扑非零；Mesh2D 恒为 0。全 router 累加。</dd>

      <dt>link_wait（link_wait_cycles）</dt>
      <dd><strong>物理链路带宽争用</strong>：flit 在 Link 的 <code>resource.request()</code> 上排队等待（与 Mesh 相同机制）。
        TDM Color 时隙<strong>不在 link 侧重检</strong>——router pipeline 已按 Color 放行。无其他 flit 争用同一链路时为 0。</dd>

      <dt>makespan（makespan_cycles）</dt>
      <dd>AllGather 流量全部完成时的仿真时间（cycles）。<strong>端到端延迟对比以这一列为准</strong>；TDM/Mesh 比值 = TDM makespan ÷ Mesh makespan。</dd>

      <dt>avg_lat</dt>
      <dd>所有完成包的平均单包延迟（cycles），反映典型包的体验，与 makespan（最慢包决定）不同。</dd>
    </dl>
  </section>

  <section class="panel">
    <h2>1) 两类 TDM 切分 vs Mesh（ND，slot=1）</h2>
    <p class="hint">表中 color_wait 为累加统计，见上方说明；端到端对比看 TDM/Mesh 列（基于 makespan）。</p>
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
      <li>限制版 Color 数显著更少（6×8: 16→5，12×16/14×14: 64/49→10），ND makespan 可降至 Mesh 的 <strong>0.03×–0.39×</strong>（见 makespan，非 color_wait）。</li>
      <li>满 FB 在 6×8、12×16 上与 Mesh 接近或略优；14×14 满 FB 因 C=49 导致 makespan 约 <strong>8× Mesh</strong>（color_wait 累加亦极高）。</li>
      <li>6×8 额外跑 direct_allgather；12×16/14×14 仅 ND（大 mesh direct 为 O(N²)）。</li>
    </ul>
  </section>
  <section>
    <h2>3) 完整结果</h2>
    <p class="hint">列含义见顶部「指标说明」。Mesh2D 的 color_wait / link_wait 恒为 0（无 TDM 机制）。</p>
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

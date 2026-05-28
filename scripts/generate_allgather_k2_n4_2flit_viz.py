#!/usr/bin/env python3
"""Generate interactive HTML: ND AllGather on 4x4 mesh + k2_n4 TDM, 2 flits/node."""

from __future__ import annotations

import json
import sys
from math import ceil
from pathlib import Path

import simpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsesim.network.collective import _groups_by_dimension, generate_collective_traffic
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly

ROWS = 4
COLS = 4
K = 2
N = 4
FLIT_BYTES = 128
CHUNK_BYTES = 256  # 每包 / 每节点本地数据：256 B = 2 flit
LINK_BW_FLITS_PER_CYCLE = 1  # 128 B/cycle
LINK_LATENCY_CYCLES = 1  # router 间链路传播时延
SLOT_CYCLES = 1
ROUTER_CYCLES_PER_FLIT = 5  # RC+VA+SA+ST(1)+crossbar(1)
TX_CYCLES_PER_FLIT = 1
OUT = Path("docs/allgather_k2_n4_2flit_viz.html")

FLIT_HOP_CYCLES = ROUTER_CYCLES_PER_FLIT + LINK_LATENCY_CYCLES + TX_CYCLES_PER_FLIT


def _build_model() -> dict:
    topo = TDMFlatButterfly(k=K, n=N, rows=ROWS, cols=COLS)
    plan = topo.coloring()
    nodes = list(range(ROWS * COLS))
    groups = _groups_by_dimension(nodes, K, N)
    payload = CHUNK_BYTES * len(nodes)
    traffic = generate_collective_traffic(
        algorithm="nd_dimension_exchange_allgather",
        participating_nodes_global=nodes,
        cores_per_reticle=len(nodes),
        payload_bytes_per_expert=payload,
        num_experts=1,
        topology_hint={"k": K, "n": N},
    )

    def enrich(item: dict) -> dict:
        src, dst = int(item["src_core"]), int(item["dst_core"])
        stage = int(item["delay_cycles"])
        hops = topo.dim_order_route(src, dst)
        logical = []
        phys: list[tuple[int, int]] = []
        colors: list[int] = []
        for u, v in hops:
            c = plan.color_of_logical[(u, v)]
            colors.append(c)
            logical.append({"u": u, "v": v, "color": c, "dim": _hop_dim(topo, u, v)})
            phys.extend(topo.physical_path(u, v))
        phys_dedup = list(dict.fromkeys(phys))
        return {
            "src": src,
            "dst": dst,
            "stage": stage,
            "chunk_bytes": int(item["size_bytes"]),
            "flits": max(1, ceil(int(item["size_bytes"]) / FLIT_BYTES)),
            "logical": logical,
            "phys": [{"u": a, "v": b} for a, b in phys_dedup],
            "mono_color": len(set(colors)) == 1,
            "colors": colors,
        }

    packets = [enrich(t) for t in traffic]
    stages = []
    for dim in range(N):
        pkts = [p for p in packets if p["stage"] == dim]
        phys_lens = {len(p["phys"]) for p in pkts}
        stage_colors = sorted({h["color"] for p in pkts for h in p["logical"]})
        stages.append(
            {
                "dim": dim,
                "inject_cycle": dim,
                "groups": groups[dim],
                "packets": pkts,
                "num_packets": len(pkts),
                "phys_hops": sorted(phys_lens),
                "colors_used": stage_colors,
            }
        )

    coords = {node: list(topo.to_coords(node)) for node in nodes}
    inject_flits = sum(p["flits"] for p in packets)
    link_flits_static = sum(p["flits"] * len(p["phys"]) for p in packets)
    sim = _run_simulation(payload, traffic, topo)
    return {
        "k": K,
        "n": N,
        "rows": ROWS,
        "cols": COLS,
        "C": plan.C,
        "flit_bytes": FLIT_BYTES,
        "chunk_bytes": CHUNK_BYTES,
        "flits_per_packet": max(1, ceil(CHUNK_BYTES / FLIT_BYTES)),
        "node_data_bytes": CHUNK_BYTES,
        "payload_bytes": payload,
        "link_latency_cycles": LINK_LATENCY_CYCLES,
        "link_bw_flits_per_cycle": LINK_BW_FLITS_PER_CYCLE,
        "total_packets": len(packets),
        "inject_flits": inject_flits,
        "link_flits_static": link_flits_static,
        "sim": sim,
        "coords": coords,
        "stages": stages,
    }


def _hop_dim(topo: TDMFlatButterfly, u: int, v: int) -> int:
    uc, vc = topo.to_coords(u), topo.to_coords(v)
    for dim in range(topo.n):
        if uc[dim] != vc[dim]:
            return dim
    return -1


def _run_simulation(payload: int, traffic: list[dict], topo: TDMFlatButterfly) -> dict:
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=topo,
        routing=TDMFlatButterflyRouting(topology=topo),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=ROWS * COLS,
        link_bw_flits_per_cycle=LINK_BW_FLITS_PER_CYCLE,
        link_latency_cycles=LINK_LATENCY_CYCLES,
        num_vcs=2,
        buffer_depth=8,
        slot_cycles=SLOT_CYCLES,
        flit_bytes=FLIT_BYTES,
    )
    flits_per_pkt = max(1, ceil(CHUNK_BYTES / FLIT_BYTES))
    records: list[dict] = []

    for item in traffic:
        delay = int(item.get("delay_cycles", 0))
        src, dst = int(item["src_core"]), int(item["dst_core"])
        stage = delay
        phys_hops = len(
            list(
                dict.fromkeys(
                    e
                    for u, v in topo.dim_order_route(src, dst)
                    for e in topo.physical_path(u, v)
                )
            )
        )

        def _inject(sim_env: simpy.Environment, delay_cycles: int, pkt: dict, meta: dict):
            if delay_cycles > 0:
                yield sim_env.timeout(delay_cycles)
            t0 = sim_env.now
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
            records.append(
                {
                    **meta,
                    "latency": int(sim_env.now - t0),
                    "done_at": int(sim_env.now),
                }
            )

        env.process(
            _inject(
                env,
                delay,
                item,
                {"stage": stage, "src": src, "dst": dst, "phys_hops": phys_hops},
            )
        )

    env.run()

    stage_stats = []
    for dim in range(N):
        recs = [r for r in records if r["stage"] == dim]
        hop_set = sorted({r["phys_hops"] for r in recs})
        phys_hops = hop_set[0] if len(hop_set) == 1 else max(hop_set)
        lats = [r["latency"] for r in recs]
        unc_lat = phys_hops * flits_per_pkt * FLIT_HOP_CYCLES
        stage_stats.append(
            {
                "dim": dim,
                "inject_cycle": dim,
                "phys_hops": phys_hops,
                "num_packets": len(recs),
                "min_latency": min(lats),
                "max_latency": max(lats),
                "last_done": max(r["done_at"] for r in recs),
                "wall_clock": dim + max(lats),
                "uncontended_latency": unc_lat,
                "uncontended_wall": dim + unc_lat,
                "contention_overhead": max(lats) - unc_lat,
            }
        )

    bottleneck = max(records, key=lambda r: r["done_at"])
    cum_hops = sum(s["phys_hops"] for s in stage_stats)

    return {
        "slot_cycles": SLOT_CYCLES,
        "link_latency_cycles": LINK_LATENCY_CYCLES,
        "link_bw_flits_per_cycle": LINK_BW_FLITS_PER_CYCLE,
        "flit_bytes": FLIT_BYTES,
        "flit_hop_cycles": FLIT_HOP_CYCLES,
        "flits_per_packet": flits_per_pkt,
        "makespan_cycles": int(env.now),
        "avg_latency": round(float(net.stats.avg_latency()), 2),
        "max_packet_latency": int(net.stats.max_packet_latency),
        "link_flits_sent": int(net.stats.flits_sent),
        "color_buffer_wait_cycles": int(net.stats.color_buffer_wait_cycles),
        "link_wait_cycles": int(net.stats.link_wait_cycles),
        "vc_wait_cycles": int(net.stats.vc_wait_cycles),
        "router_pipeline_cycles": int(net.stats.pipeline_cycles),
        "stage_stats": stage_stats,
        "cumulative_phys_hops": cum_hops,
        "uncontended_makespan": max(s["uncontended_wall"] for s in stage_stats),
        "bottleneck": {
            "stage": bottleneck["stage"],
            "src": bottleneck["src"],
            "dst": bottleneck["dst"],
            "phys_hops": bottleneck["phys_hops"],
            "latency": bottleneck["latency"],
            "done_at": bottleneck["done_at"],
        },
    }


def _html(model: dict) -> str:
    data = json.dumps(model, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>4×4 Mesh · k2_n4 TDM · ND AllGather · 256 B/包</title>
<style>
  :root {{
    --bg:#0f1419; --panel:#1a2332; --text:#e8edf4; --muted:#94a3b8;
    --border:#2d3a4f; --mesh:#334155; --c0:#2563eb; --c1:#ea580c;
    --accent:#60a5fa; --good:#4ade80;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
    background:var(--bg); color:var(--text); line-height:1.55; }}
  header {{ padding:2rem; background:linear-gradient(135deg,#1e293b,#0f172a 60%,#1a2e1a);
    border-bottom:1px solid var(--border); }}
  header h1 {{ margin:0 0 .5rem; font-size:1.65rem; }}
  header p {{ margin:.2rem 0; color:var(--muted); max-width:54rem; }}
  main {{ max-width:1180px; margin:0 auto; padding:1.5rem 1.25rem 3rem; }}
  section {{ margin-bottom:2rem; }}
  h2 {{ font-size:1.15rem; margin:0 0 .75rem; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.85rem; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:.9rem 1rem; }}
  .stat .val {{ font-size:1.5rem; font-weight:700; }}
  .stat .val.hero {{ font-size:2rem; color:var(--good); }}
  .stat .lbl {{ color:var(--muted); font-size:.82rem; }}
  .makespan-banner {{ margin-top:1rem; padding:1rem 1.25rem; border-radius:10px;
    background:linear-gradient(90deg,rgba(74,222,128,.12),rgba(96,165,250,.08));
    border:1px solid rgba(74,222,128,.35); }}
  .makespan-banner .big {{ font-size:2.2rem; font-weight:800; color:var(--good); line-height:1.2; }}
  .makespan-banner .sub {{ color:var(--muted); font-size:.88rem; margin-top:.35rem; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:1rem 1.15rem; }}
  .node-ref {{ font-family:ui-monospace,monospace; background:#111827; border:1px solid var(--border);
    border-radius:8px; padding:.85rem 1rem; white-space:pre; font-size:.88rem; }}
  .tdm-bar {{ display:flex; gap:6px; max-width:360px; margin:.75rem 0; }}
  .tdm-slot {{ flex:1; text-align:center; padding:.65rem .4rem; border-radius:8px; font-size:.82rem; font-weight:600; }}
  .tdm-slot.s0 {{ background:rgba(37,99,235,.22); color:#93c5fd; }}
  .tdm-slot.s1 {{ background:rgba(234,88,12,.22); color:#fdba74; }}
  .stage-tabs {{ display:flex; gap:.45rem; flex-wrap:wrap; margin-bottom:.85rem; }}
  .stage-tab {{ border:1px solid var(--border); background:#243044; color:var(--text);
    padding:.45rem .85rem; border-radius:7px; cursor:pointer; font-size:.84rem; }}
  .stage-tab.active {{ background:#334155; border-color:#64748b; }}
  .view-tabs {{ display:flex; gap:.45rem; margin:.5rem 0 .75rem; }}
  .view-tab {{ border:1px solid var(--border); background:#1e293b; color:var(--muted);
    padding:.35rem .7rem; border-radius:6px; cursor:pointer; font-size:.8rem; }}
  .view-tab.active {{ color:var(--text); background:#334155; }}
  .layout {{ display:grid; grid-template-columns:1fr 320px; gap:1rem; align-items:start; }}
  @media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} }}
  .svg-wrap {{ background:#111827; border-radius:8px; padding:.35rem; overflow:auto; }}
  .topo-svg {{ width:100%; max-width:520px; height:auto; display:block; margin:0 auto; }}
  .mesh-edge {{ stroke:var(--mesh); stroke-width:1.2; opacity:.35; }}
  .phys-edge {{ stroke-width:3; opacity:.85; stroke-linecap:round; }}
  .phys-edge.c0 {{ stroke:var(--c0); }}
  .phys-edge.c1 {{ stroke:var(--c1); }}
  .phys-edge.dimmed {{ opacity:.08; }}
  .phys-edge.hi {{ opacity:1; stroke-width:4.5; filter:drop-shadow(0 0 4px currentColor); }}
  .node {{ fill:#1e293b; stroke:#64748b; stroke-width:1.5; }}
  .node.src {{ stroke:#4ade80; stroke-width:2.5; }}
  .node.dst {{ stroke:#f472b6; stroke-width:2.5; }}
  .node-label {{ fill:#e2e8f0; font-size:12px; font-family:ui-monospace,monospace; text-anchor:middle; }}
  .flit-badge {{ font-size:9px; fill:#cbd5e1; font-family:ui-monospace,monospace; }}
  .side {{ font-size:.86rem; color:var(--muted); }}
  .side strong {{ color:var(--text); }}
  table {{ width:100%; border-collapse:collapse; font-size:.84rem; margin-top:.5rem; }}
  th,td {{ border:1px solid var(--border); padding:.45rem .55rem; text-align:left; }}
  th {{ background:#243044; }}
  .legend {{ display:flex; gap:1rem; flex-wrap:wrap; font-size:.82rem; color:var(--muted); margin-top:.5rem; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .timeline {{ display:flex; gap:4px; align-items:stretch; margin:.75rem 0; overflow-x:auto; }}
  .tl-stage {{ min-width:88px; border:1px solid var(--border); border-radius:8px; padding:.5rem;
    background:#111827; font-size:.78rem; cursor:pointer; }}
  .tl-stage.active {{ border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }}
  .tl-stage .dim {{ font-weight:700; color:var(--accent); }}
  code {{ background:#243044; padding:.1rem .35rem; border-radius:4px; font-size:.84em; }}
  footer {{ text-align:center; color:var(--muted); font-size:.78rem; padding:1.5rem; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <h1>4×4 物理 Mesh · 2-ary 4-flat TDM · ND AllGather（256 B/包）</h1>
  <p>物理拓扑：4×4 Mesh2D（16 PE）。逻辑 overlay：<code>k=2, n=4</code> Flattened Butterfly，<code>C=2</code> Color TDM。
     集合通信：<code>nd_dimension_exchange_allgather</code>；每节点本地 256 B，每 stage 交换包 256 B（2 flit × 128 B）。
     链路：<strong>128 B/cycle</strong>（1 flit/cycle），router 间时延 <strong>1 cycle/hop</strong>。</p>
  <div class="makespan-banner" id="makespan-banner"></div>
</header>
<main>
  <section>
    <div class="grid" id="summary-grid"></div>
  </section>

  <section class="panel">
    <h2>链路模型</h2>
    <table>
      <thead><tr><th>参数</th><th>值</th><th>说明</th></tr></thead>
      <tbody>
        <tr><td>包大小</td><td><strong>256 B</strong></td><td>2 flit × 128 B/flit</td></tr>
        <tr><td>链路带宽</td><td><strong>128 B/cycle</strong></td><td><code>link_bw_flits_per_cycle=1</code></td></tr>
        <tr><td>链路时延</td><td><strong>1 cycle/hop</strong></td><td><code>link_latency_cycles=1</code>；单 flit 过链 1+1=2 cyc</td></tr>
        <tr><td>flit 粒度</td><td>128 B</td><td>网络按 flit 流水传输，每包 2 flit</td></tr>
      </tbody>
    </table>
  </section>

  <section class="panel" id="breakdown-panel">
    <h2>Makespan 分解（<span id="ms-val"></span> cycles）</h2>
    <p class="side" id="six-hop-note"></p>
    <table id="breakdown-table">
      <thead>
        <tr>
          <th>Stage</th><th>inject</th><th>phys hop</th><th>无争用 latency</th>
          <th>实测 max latency</th><th>争用开销</th><th>stage 完成时刻</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
    <table style="margin-top:1rem">
      <thead><tr><th>对比项</th><th>cycles</th><th>说明</th></tr></thead>
      <tbody id="theory-rows"></tbody>
    </table>
  </section>

  <section class="panel" id="sim-panel">
    <h2>SimPy 端到端仿真（slot_cycles=<span id="slot-label"></span>，link_latency=<span id="link-lat-label"></span>）</h2>
    <table id="sim-table">
      <thead><tr><th>指标</th><th>值</th><th>说明</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>

  <section class="panel">
    <h2>节点编号 & 逻辑坐标 (d₃,d₂,d₁,d₀)</h2>
    <div class="node-ref"> 0   1   2   3     node = d₀ + 2·d₁ + 4·d₂ + 8·d₃
 4   5   6   7
 8   9  10  11
12  13  14  15</div>
    <p class="side" style="margin-top:.65rem">例：node 5 = (1,0,1,0)；node 10 = (0,1,0,1)。ND 算法在每一维 d 上，按其余坐标相同的 <code>k=2</code> 组内成对交换。</p>
  </section>

  <section class="panel">
    <h2>Color TDM 时隙</h2>
    <div class="tdm-bar">
      <div class="tdm-slot s0">t ≡ 0 (mod 2)<br>Color 0 · 蓝</div>
      <div class="tdm-slot s1">t ≡ 1 (mod 2)<br>Color 1 · 橙</div>
    </div>
    <p class="side">每条逻辑 FB 链路固定绑定一个 Color；flit 在 router 出口等待 <code>current_color(t) == flit.color</code> 后才进入物理链路。
       示意图用<strong>物理 mesh 实线箭头</strong>表示 XY 路由，颜色 = 该逻辑 hop 的 TDM Color。</p>
  </section>

  <section class="panel">
    <h2>ND AllGather 四阶段时间线</h2>
    <p class="side">每 stage 注入 offset = dim（cycle 0/1/2/3）。组内双向交换：每对节点各发 1 包 × 256 B。</p>
    <div class="timeline" id="timeline"></div>
  </section>

  <section class="panel">
    <h2>Stage 可视化</h2>
    <div class="stage-tabs" id="stage-tabs"></div>
    <div class="view-tabs">
      <button class="view-tab active" data-view="all">全部 16 包</button>
      <button class="view-tab" data-view="sample">示例组 0↔1</button>
      <button class="view-tab" data-view="phys">物理 hop 热力</button>
    </div>
    <div class="layout">
      <div class="svg-wrap" id="main-svg"></div>
      <div class="side" id="side-info"></div>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:var(--c0)"></span> Color 0 物理链路</span>
      <span><span class="dot" style="background:var(--c1)"></span> Color 1 物理链路</span>
      <span><span class="dot" style="background:#4ade80"></span> 源节点</span>
      <span><span class="dot" style="background:#f472b6"></span> 目的节点</span>
    </div>
    <table id="pkt-table">
      <thead><tr><th>src→dst</th><th>逻辑 hop</th><th>Color</th><th>物理 hop 数</th><th>flit</th></tr></thead>
      <tbody></tbody>
    </table>
  </section>

  <section class="panel">
    <h2>数据块演化（概念）</h2>
    <table>
      <thead><tr><th>Stage 后</th><th>每节点已知 chunk 数</th><th>说明</th></tr></thead>
      <tbody>
        <tr><td>初始</td><td>1 × 256 B</td><td>节点 i 持有本地 chunkᵢ</td></tr>
        <tr><td>dim 0</td><td>2</td><td>同组 d₀ 邻居互换 → 覆盖 2 个 d₀ 位</td></tr>
        <tr><td>dim 1</td><td>4</td><td>再沿 d₁ 扩展</td></tr>
        <tr><td>dim 2</td><td>8</td><td>再沿 d₂ 扩展</td></tr>
        <tr><td>dim 3</td><td>16 × 256 B</td><td>全收集完成：每节点 4096 B</td></tr>
      </tbody>
    </table>
  </section>
</main>
<footer>生成自 scripts/generate_allgather_k2_n4_2flit_viz.py · wse-simulator</footer>

<script>
const MODEL = {data};
const ROWS = MODEL.rows, COLS = MODEL.cols, CELL = 105, MARGIN = 62, W = 460, H = 460;
let curStage = 0, curView = 'all', hoverKey = null;

function nodeXY(n) {{
  const r = Math.floor(n / COLS), c = n % COLS;
  return [MARGIN + c * CELL, MARGIN + r * CELL];
}}

function meshSkeleton() {{
  let s = '';
  for (let n = 0; n < ROWS * COLS; n++) {{
    const [x0,y0] = nodeXY(n);
    const r = Math.floor(n/COLS), c = n % COLS;
    if (c+1 < COLS) {{ const [x1,y1]=nodeXY(n+1); s += `<line x1="${{x0}}" y1="${{y0}}" x2="${{x1}}" y2="${{y1}}" class="mesh-edge"/>`; }}
    if (r+1 < ROWS) {{ const [x1,y1]=nodeXY(n+COLS); s += `<line x1="${{x0}}" y1="${{y0}}" x2="${{x1}}" y2="${{y1}}" class="mesh-edge"/>`; }}
  }}
  return s;
}}

function edgeKey(a,b) {{ return a < b ? `${{a}}-${{b}}` : `${{b}}-${{a}}`; }}

function filterPackets(stage, view) {{
  const pkts = stage.packets;
  if (view === 'sample') {{
    const g = stage.groups[0];
    const set = new Set(g);
    return pkts.filter(p => set.has(p.src) && set.has(p.dst));
  }}
  return pkts;
}}

function buildSvg(stage, view) {{
  const pkts = filterPackets(stage, view);
  const edgeUse = {{}};
  const nodeRole = {{}};
  for (const p of pkts) {{
    nodeRole[p.src] = 'src';
    nodeRole[p.dst] = 'dst';
    for (const e of p.phys) {{
      const k = edgeKey(e.u, e.v);
      edgeUse[k] = (edgeUse[k] || 0) + 1;
    }}
  }}
  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" class="topo-svg">`;
  svg += meshSkeleton();
  // physical edges
  if (view === 'phys') {{
    const maxLoad = Math.max(1, ...Object.values(edgeUse), 1);
    for (const [k, load] of Object.entries(edgeUse)) {{
      const [a,b] = k.split('-').map(Number);
      const [x0,y0]=nodeXY(a), [x1,y1]=nodeXY(b);
      const op = 0.35 + 0.65 * (load / maxLoad);
      svg += `<line x1="${{x0}}" y1="${{y0}}" x2="${{x1}}" y2="${{y1}}" stroke="#60a5fa" stroke-width="${{2+3*load/maxLoad}}" opacity="${{op}}"/>`;
      const mx=(x0+x1)/2, my=(y0+y1)/2;
      svg += `<text x="${{mx}}" y="${{my}}" fill="#e2e8f0" font-size="10" text-anchor="middle">${{load}}</text>`;
    }}
  }} else {{
    const drawn = new Set();
    pkts.forEach((p, pi) => {{
      p.phys.forEach((e, ei) => {{
        const k = `${{p.src}}-${{p.dst}}-${{e.u}}-${{e.v}}`;
        if (drawn.has(k)) return;
        drawn.add(k);
        const color = p.logical[0]?.color ?? 0;
        const cls = `phys-edge c${{color}}`;
        const [x0,y0]=nodeXY(e.u), [x1,y1]=nodeXY(e.v);
        const hi = hoverKey === `${{p.src}}-${{p.dst}}`;
        svg += `<line x1="${{x0}}" y1="${{y0}}" x2="${{x1}}" y2="${{y1}}" class="${{cls}} ${{hi?'hi':''}}" data-pkt="${{p.src}}-${{p.dst}}" marker-end="url(#arrow-c${{color}})"/>`;
      }});
    }});
    svg += `<defs>
      <marker id="arrow-c0" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#2563eb"/></marker>
      <marker id="arrow-c1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#ea580c"/></marker>
    </defs>`;
  }}
  for (let n = 0; n < ROWS*COLS; n++) {{
    const [x,y] = nodeXY(n);
    const role = nodeRole[n] || '';
    svg += `<circle cx="${{x}}" cy="${{y}}" r="17" class="node ${{role}}"/>`;
    svg += `<text x="${{x}}" y="${{y+4}}" class="node-label">${{n}}</text>`;
    if (curStage === 0 && view === 'sample' && (n===0||n===1))
      svg += `<text x="${{x}}" y="${{y+16}}" class="flit-badge">256B</text>`;
  }}
  svg += '</svg>';
  return svg;
}}

function renderSide(stage, view) {{
  const pkts = filterPackets(stage, view);
  const hops = stage.phys_hops.join(' / ');
  const colors = stage.colors_used.map(c => `C${{c}}`).join(', ');
  let html = `<p><strong>Stage dim ${{stage.dim}}</strong> · inject@${{stage.inject_cycle}} · ${{stage.num_packets}} 包 · 物理 hop: ${{hops}}</p>`;
  html += `<p>使用 Color: <strong>${{colors}}</strong></p>`;
  html += `<p>组数: ${{stage.groups.length}} × k=${{MODEL.k}} = ${{stage.num_packets}} 条有向流</p>`;
  if (view === 'sample') {{
    const g = stage.groups[0];
    html += `<p>示例组 <code>[${{g.join(', ')}}]</code>：双向各发 256 B（2 flit）</p>`;
    html += `<p>0→1: 逻辑 dim0, Color 1, 1 物理 hop<br>1→0: 对称</p>`;
  }}
  if (stage.dim === 1 && view === 'sample') {{
    html += `<p>0→2: 逻辑 dim1, Color 0, 物理 0→1→2（2 hop）</p>`;
  }}
  document.getElementById('side-info').innerHTML = html;
}}

function renderTable(stage, view) {{
  const pkts = filterPackets(stage, view);
  const tbody = document.querySelector('#pkt-table tbody');
  tbody.innerHTML = pkts.map(p => {{
    const log = p.logical.map(h => `${{h.u}}→${{h.v}}(d${{h.dim}})`).join(' ');
    const col = p.mono_color ? `C${{p.colors[0]}}` : p.colors.map(c=>'C'+c).join('+');
    return `<tr data-pkt="${{p.src}}-${{p.dst}}"><td>${{p.src}}→${{p.dst}}</td><td>${{log}}</td><td>${{col}}</td><td>${{p.phys.length}}</td><td>${{p.flits}}</td></tr>`;
  }}).join('');
  tbody.querySelectorAll('tr').forEach(tr => {{
    tr.addEventListener('mouseenter', () => {{ hoverKey = tr.dataset.pkt; renderStage(); }});
    tr.addEventListener('mouseleave', () => {{ hoverKey = null; renderStage(); }});
  }});
}}

function renderStage() {{
  const stage = MODEL.stages[curStage];
  document.getElementById('main-svg').innerHTML = buildSvg(stage, curView);
  renderSide(stage, curView);
  renderTable(stage, curView);
  document.querySelectorAll('.stage-tab').forEach((btn,i) => btn.classList.toggle('active', i===curStage));
  document.querySelectorAll('.tl-stage').forEach((el,i) => el.classList.toggle('active', i===curStage));
  document.querySelectorAll('.view-tab').forEach(btn => btn.classList.toggle('active', btn.dataset.view===curView));
}}

function init() {{
  const sim = MODEL.sim;
  const fh = sim.flit_hop_cycles;
  const fp = sim.flits_per_packet;
  document.getElementById('makespan-banner').innerHTML =
    `<div class="big">Makespan = ${{sim.makespan_cycles}} cycles</div>` +
    `<div class="sub">256 B/包 · 128 B/cycle · link_latency=${{sim.link_latency_cycles}} cyc/hop · ` +
    `flit/hop=${{fh}} cyc · 累计 ${{sim.cumulative_phys_hops}} phys hops（4 stage）· ` +
    `无争用下界 ${{sim.uncontended_makespan}} cyc</div>`;

  document.getElementById('ms-val').textContent = sim.makespan_cycles;
  document.getElementById('six-hop-note').innerHTML =
    `<strong>6 hop 是对的，但不等于 makespan。</strong> ND 4 阶段每包物理 hop 为 1+2+1+2 = <strong>${{sim.cumulative_phys_hops}}</strong> ` +
    `（同一 chunk 若依次参与 4 次交换的累计跳数）。但 makespan = max<sub>stage</sub>(inject + 该 stage 最慢包 latency)，` +
    `4 个 stage 并行重叠，不是把 6 hop 串成一条路径。每 stage 发完整 256 B（${{fp}} flit），` +
    `单 flit 单 hop 代价 = router(5) + link(1+1) = ${{fh}} cyc。`;

  document.querySelector('#breakdown-table tbody').innerHTML = sim.stage_stats.map(s => {{
    const hot = s.last_done === sim.makespan_cycles;
    return `<tr${{hot?' style="outline:1px solid var(--good)"':''}}>` +
      `<td>dim ${{s.dim}}</td><td>@${{s.inject_cycle}}</td><td>${{s.phys_hops}}</td>` +
      `<td>${{s.uncontended_latency}}</td><td><strong>${{s.max_latency}}</strong></td>` +
      `<td>+${{s.contention_overhead}}</td><td>${{s.last_done}}</td></tr>`;
  }}).join('');

  const bn = sim.bottleneck;
  const theory = [
    ['单 flit 单 hop', fh, 'router 5 + link(1+1)'],
    ['256 B 单包 1-hop', fp * fh, `${{fp}} flit × ${{fh}}（dim0/2）`],
    ['256 B 单包 2-hop 无争用', 2 * fp * fh, `${{fp}} flit × 2 hop × ${{fh}}（dim1/3）`],
    ['6 hop × 单 flit（串行假想）', sim.cumulative_phys_hops * fh, '若 1 flit 连续走 6 hop，仍非 ND 调度模型'],
    ['6 hop × 256 B（串行假想）', sim.cumulative_phys_hops * fp * fh, '4 次交换各走 1/2 hop，不是一条 6-hop 路由'],
    ['无争用 makespan 下界', sim.uncontended_makespan, 'max(inject + 无争用 latency) = max(14,29,16,31)'],
    ['实测 makespan', sim.makespan_cycles, `瓶颈 dim${{bn.stage}} ${{bn.src}}→${{bn.dst}} inject@${{bn.stage}} lat=${{bn.latency}} → ${{bn.done_at}}`],
    ['争用额外开销', sim.makespan_cycles - sim.uncontended_makespan, '主要来自 dim1/3 的 16 路 2-hop 并发 + VC=2'],
  ];
  document.getElementById('theory-rows').innerHTML = theory.map(([a,b,c]) =>
    `<tr><td>${{a}}</td><td><strong>${{b}}</strong></td><td>${{c}}</td></tr>`
  ).join('');

  const grid = document.getElementById('summary-grid');
  const stats = [
    ['Makespan', `${{sim.makespan_cycles}} cyc`, '端到端墙钟'],
    ['包大小', '256 B', '2 flit × 128 B'],
    ['链路时延', `${{sim.link_latency_cycles}} cyc`, 'router 间/hop'],
    ['链路带宽', '128 B/cyc', '1 flit/cycle'],
    ['注入 flit', String(MODEL.inject_flits), '64 包 × 2 flit'],
    ['TDM C', String(MODEL.C), 'Color 周期'],
  ];
  grid.innerHTML = stats.map(([lbl,val]) =>
    `<div class="stat"><div class="val ${{lbl==='Makespan'?'hero':''}}">${{val}}</div><div class="lbl">${{lbl}}</div></div>`
  ).join('');

  document.getElementById('slot-label').textContent = sim.slot_cycles;
  document.getElementById('link-lat-label').textContent = sim.link_latency_cycles;
  const simRows = [
    ['makespan_cycles', sim.makespan_cycles, '最后一个包完成时刻（墙钟）'],
    ['link_latency_cycles', sim.link_latency_cycles, 'router 间链路传播时延/hop'],
    ['link_bw', `${{sim.link_bw_flits_per_cycle}} flit/cyc (${{sim.flit_bytes}} B/cyc)`, '链路传输带宽'],
    ['avg_latency', sim.avg_latency, '所有包 latency 均值'],
    ['max_packet_latency', sim.max_packet_latency, '最慢包（瓶颈路径）'],
    ['link_flits_sent', sim.link_flits_sent, '链路级 flit 传输总次数'],
    ['color_buffer_wait_cycles', sim.color_buffer_wait_cycles, 'TDM color 门控等待（全网累加）'],
    ['link_wait_cycles', sim.link_wait_cycles, '链路排队等待（全网累加）'],
    ['router_pipeline_cycles', sim.router_pipeline_cycles, 'router pipeline（全网累加）'],
  ];
  document.querySelector('#sim-table tbody').innerHTML = simRows.map(([k,v,d]) =>
    `<tr><td><code>${{k}}</code></td><td><strong>${{v}}</strong></td><td>${{d}}</td></tr>`
  ).join('');

  const tabs = document.getElementById('stage-tabs');
  MODEL.stages.forEach((s,i) => {{
    const b = document.createElement('button');
    b.className = 'stage-tab' + (i===0?' active':'');
    b.textContent = `dim ${{s.dim}} (${{s.num_packets}} pkts, ${{s.phys_hops.join('/')}} hop)`;
    b.onclick = () => {{ curStage = i; renderStage(); }};
    tabs.appendChild(b);
  }});
  const tl = document.getElementById('timeline');
  MODEL.stages.forEach((s,i) => {{
    const d = document.createElement('div');
    d.className = 'tl-stage' + (i===0?' active':'');
    d.innerHTML = `<div class="dim">dim ${{s.dim}}</div>inject@${{s.inject_cycle}}<br>${{s.num_packets}}×256 B<br>${{s.phys_hops.join('/')}} phys hop`;
    d.onclick = () => {{ curStage = i; renderStage(); }};
    tl.appendChild(d);
  }});
  document.querySelectorAll('.view-tab').forEach(btn => {{
    btn.onclick = () => {{ curView = btn.dataset.view; renderStage(); }};
  }});
  renderStage();
}}
init();
</script>
</body>
</html>
"""


def main() -> None:
    model = _build_model()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(_html(model), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

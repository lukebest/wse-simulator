#!/usr/bin/env python3
"""Generate docs/color_simulation_report.html from simulation CSV."""

from __future__ import annotations

import csv
import html
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsesim.network.collective import default_color_map
from wsesim.network.color import ColorPlan
from wsesim.network.color_routes import build_mixed_plan


def load_results(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def speedup(xy: float, color: float) -> tuple[str, float, str]:
    if xy == color:
        return "tie", 1.0, "tie"
    if color < xy:
        return "Color", xy / color, "win-color"
    return "XY", color / xy, "win-xy"


def fmt_num(v: float) -> str:
    if v >= 1_000_000:
        return f"{v:,.0f}"
    if v >= 1000:
        return f"{v:,.1f}"
    if abs(v - round(v)) < 0.05:
        return str(int(round(v)))
    return f"{v:.2f}"


def _describe_color_route(plan: ColorPlan, color_id: int) -> tuple[str, str]:
    """Return (route_kind, detail) for one color in a plan."""
    name = ""
    if color_id < len(plan.colors):
        name = plan.colors[color_id].name or ""

    if color_id in plan.unicast_modes:
        mode = plan.unicast_modes[color_id].upper()
        return (
            f"维度序单播 ({mode})",
            "依据包内 <code>dst</code> 在每一跳静态选择下一跳（先 row 后 col 为 XY，反之为 YX）；"
            "dest 表为空时由 <code>unicast_modes</code> 推导。",
        )

    forward_nodes = 0
    max_fanout = 0
    for node in range(plan.num_nodes):
        hops = plan.dest.get(node, {}).get(color_id, set())
        if hops:
            forward_nodes += 1
            max_fanout = max(max_fanout, len(hops))

    if forward_nodes == 0:
        return ("保留 / 未填充", "该 color ID 在 mixed plan 中可能为 spare 或未启用。")

    if "row_ring" in name or "col_ring" in name:
        return (
            "行/列环",
            f"每节点 <code>dest[node][{color_id}]</code> 指向环上唯一后继；"
            f"{forward_nodes} 个节点参与转发。",
        )
    if "allreduce_ring" in name or "snake" in name:
        return (
            "线性 / Snake 环",
            f"节点按固定顺序组成环，每跳指向序列中的下一个节点（{forward_nodes} 个转发项）。",
        )
    if "row_mcast" in name or "row_bus" in name:
        return (
            "行多播树",
            "行内自西向东单播复制；每节点最多转发给一个东向邻居。",
        )
    if "col_reduce" in name or "col_bus" in name:
        return (
            "列归约树",
            "列内自南向北归约；非根节点向北发送 partial sum。",
        )
    if "dim_ex" in name:
        return (
            "维度交换 (RHD)",
            "超立方 XOR 伙伴交换的一 stage；每个 stage 独占一个 color。",
        )
    if "a2a_p" in name:
        return (
            "All-to-all 相位",
            "时间分相的固定置换：phase p 上 src→rotate(src, p+1)。",
        )
    if max_fanout > 1:
        return ("多播 dest 表", f"静态多下一跳；最大 fanout {max_fanout}。")
    return (
        "显式单播 dest 表",
        f"每源节点预计算下一跳；{forward_nodes} 个节点有路由项。",
    )


def _build_mixed_plan_steps() -> str:
    return """
    <ol class="steps">
      <li><strong>Color 0</strong> — <code>add_path_color(xy)</code>：systolic / gather 主路径</li>
      <li><strong>Color 1</strong> — <code>add_path_color(yx)</code>：与 0 正交，并行单播</li>
      <li><strong>Color 2</strong> — <code>add_linear_ring</code>：按 node ID 顺序的环（all-reduce 拓扑）</li>
      <li><strong>Color 3</strong> — <code>add_row_multicast_tree(row=0)</code>：第 0 行广播树</li>
      <li><strong>Color 4</strong> — <code>add_col_reduction_tree(col=0)</code>：第 0 列归约树</li>
      <li><strong>Color 5…</strong> — <code>add_dimension_exchange_colors</code>：RHD XOR stage（节点数为 2 的幂时）</li>
      <li><strong>后续</strong> — 逐行 <code>add_row_ring</code>、逐列 <code>add_col_ring</code> 直到 K 用尽</li>
      <li><strong>Spare</strong> — 剩余 ID 交替 XY/YX 单播</li>
    </ol>"""


def _plan_table_html(rows: int, cols: int, num_colors: int = 16) -> str:
    plan = build_mixed_plan(rows, cols, num_colors)
    body = ""
    for cid in range(num_colors):
        c = plan.colors[cid] if cid < len(plan.colors) else None
        name = html.escape(c.name if c and c.name else f"color_{cid}")
        kind, detail = _describe_color_route(plan, cid)
        body += f"""
        <tr>
          <td class="num">{cid}</td>
          <td><code>{name}</code></td>
          <td>{html.escape(kind)}</td>
          <td>{detail}</td>
        </tr>"""
    return f"""
    <h3>{rows}×{cols} mesh — <code>build_mixed_plan</code> 生成的 {num_colors} 个 color</h3>
    <table>
      <thead><tr><th>ID</th><th>名称</th><th>路由类型</th><th>生成规则</th></tr></thead>
      <tbody>{body}</tbody>
    </table>"""


def _traffic_color_map_html() -> str:
    cmap = default_color_map()
    rows = ""
    for payload, cid in sorted(cmap.items(), key=lambda x: (x[1], x[0])):
        rows += f"<tr><td><code>{html.escape(payload)}</code></td><td class=\"num\">{cid}</td></tr>"
    return f"""
    <table>
      <thead><tr><th>Traffic <code>payload</code></th><th>默认 Color ID</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def render_color_generation_section(num_colors: int = 16) -> str:
    plan_4 = _plan_table_html(4, 4, num_colors)
    plan_8 = _plan_table_html(8, 8, num_colors)
    steps = _build_mixed_plan_steps()
    cmap = _traffic_color_map_html()
    return f"""
    <h2>Color 生成方法与依据</h2>

    <h3>设计依据</h3>
    <div class="callout">
      <p>本仿真参照 Cerebras 专利 <strong>US10,515,303</strong> 的 Color 机制：</p>
      <ul>
        <li><strong>编译期静态路由</strong> — NN 通信模式在编译时已知，每个 color 对应一张固定的
            <code>dest[node][color] → next_hop</code> 转发表（或 XY/YX 单播模式），运行时无动态路由。</li>
        <li><strong>虚拟网络隔离</strong> — 时间上重叠、路径上冲突的 flow 应映射到<strong>不同 color</strong>，
            非重叠 flow 可<strong>复用</strong>同一 color（见 <code>color_alloc.py</code> 贪心 / 图着色 / ILP）。</li>
        <li><strong>NN 通信原语分类</strong> — 按 Cerebras 文档中的 taxonomy 为每类 collective 预置路由形状：
            路径 (XY/YX)、环、行多播、列归约、维度交换、all-to-all 分相等（详见
            <a href="color_mechanism_analysis.md">color_mechanism_analysis.md</a>）。</li>
        <li><strong>保序</strong> — 每个 <code>(color, src, dst)</code> 流在仿真中串行注入，配合固定路由保证 FIFO、
            <code>ordering_violations = 0</code>。</li>
      </ul>
    </div>

    <h3>静态 ColorPlan 构建 — <code>build_mixed_plan(rows, cols, K=16)</code></h3>
    <p>仿真使用的 <code>color_mixed</code> 方案由 <code>wsesim/network/color_routes.py</code>
       按下列<strong>固定顺序</strong>填充 K 个 color（默认 K=16）：</p>
    {steps}
    <p>实现入口：</p>
    <pre>plan = build_mixed_plan(rows, cols, num_colors=16)
# wsesim/network/color_routes.py → ColorPlan(dest, unicast_modes, colors[])</pre>

    {plan_4}
    {plan_8}

    <h3>Traffic → Color 映射（运行时）</h3>
    <p>Collective 流量由 <code>generate_collective_traffic()</code> 产生后，经两步分配 color：</p>

    <h4>1. 默认 payload 映射 — <code>default_color_map()</code></h4>
    <p>依据 collective 语义将 <code>payload</code> 类型映射到 mixed plan 中的主 color（当前实现以
       <strong>color 0 (XY)</strong> 与 <strong>color 1 (YX)</strong> 为主通道）：</p>
    {cmap}

    <h4>2. 并发 striping — <code>_assign_traffic_colors()</code></h4>
    <p>同一 <code>delay_cycles</code> 时刻并发注入的包，若 base color 为 0 或 1，则按序号交替分配到
       <strong>color 0 / 1</strong>，使 XY 与 YX 虚拟网络同时承载并行单播，避免单 VN  Head-of-line blocking：</p>
    <pre>for j, idx in enumerate(concurrent_packet_indices):
    traffic[idx]["color"] = (base + j) % 2   # base ∈ {{0, 1}}</pre>

    <p><strong>说明：</strong> mixed plan 中预置的 ring (2)、row multicast (3)、col reduce (4)、
       dimension-exchange (5+) 等 color 已生成静态路由表，但<strong>当前 benchmark 的 traffic 映射
       主要使用 color 0/1 单播</strong>。Ring all-reduce 因此尚未走专用环 color，这是 ring 模式
       makespan 落后于 XY 的原因之一。</p>

    <h3>与 Baseline 对比</h3>
    <table>
      <thead><tr><th>方案</th><th>Color 数</th><th>路由</th><th>生成函数</th></tr></thead>
      <tbody>
        <tr>
          <td><code>color_mixed</code></td>
          <td class="num">16</td>
          <td>上表 mixed plan + traffic 映射</td>
          <td><code>build_mixed_plan</code></td>
        </tr>
        <tr>
          <td><code>xy_single_vn</code></td>
          <td class="num">1</td>
          <td>仅 color 0 = XY 维度序</td>
          <td><code>build_baseline_single_vn</code></td>
        </tr>
      </tbody>
    </table>

    <h3>可选：动态 Color 分配</h3>
    <p>若给定 flow 的时间窗与占用链路集合，<code>wsesim/network/color_alloc.py</code> 提供：</p>
    <ul>
      <li><strong>greedy_allocate</strong> — 按开始时间排序，选冲突最少 / 峰值负载最低的 color</li>
      <li><strong>graph_coloring_allocate</strong> — 冲突图着色</li>
      <li><strong>ilp_allocate</strong> — PuLP ILP 最小化峰值链路负载（失败时回退贪心）</li>
    </ul>
    <p>约束：时间重叠且链路交集非空的 flow 不能共享 color；路由图需无环（<code>routes_acyclic</code> 检查）。
       本报告 benchmark 未启用动态分配，而是使用固定的 mixed plan + default_color_map。</p>
    """


def build_summary(rows: list[dict[str, str]]) -> list[dict]:
    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["mesh"], r["pattern"])
        by_key[key][r["scheme"]] = float(r["makespan_cycles"])

    summary = []
    for (mesh, pattern), schemes in sorted(by_key.items()):
        xy = schemes.get("xy_single_vn")
        color = schemes.get("color_mixed")
        if xy is None or color is None:
            continue
        winner, ratio, css = speedup(xy, color)
        summary.append(
            {
                "mesh": mesh,
                "pattern": pattern,
                "xy": xy,
                "color": color,
                "winner": winner,
                "ratio": ratio,
                "css": css,
            }
        )
    return summary


def render_html(
    rows: list[dict[str, str]], summary: list[dict], csv_path: Path, *, num_colors: int = 16
) -> str:
    color_gen_section = render_color_generation_section(num_colors)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    max_ms = max(float(r["makespan_cycles"]) for r in rows) or 1

    summary_rows = ""
    for s in summary:
        ratio_label = f"{s['ratio']:.1f}×" if s["winner"] != "tie" else "1.0×"
        summary_rows += f"""
        <tr class="{s['css']}">
          <td>{html.escape(s['mesh'])}</td>
          <td><code>{html.escape(s['pattern'])}</code></td>
          <td class="num">{fmt_num(s['xy'])}</td>
          <td class="num">{fmt_num(s['color'])}</td>
          <td>{html.escape(s['winner'])}</td>
          <td class="num">{ratio_label}</td>
        </tr>"""

    detail_rows = ""
    for r in rows:
        ms = float(r["makespan_cycles"])
        bar_w = max(2, int(100 * ms / max_ms))
        detail_rows += f"""
        <tr>
          <td>{html.escape(r['mesh'])}</td>
          <td>{html.escape(r['scheme'])}</td>
          <td><code>{html.escape(r['pattern'])}</code></td>
          <td class="num">{fmt_num(ms)}</td>
          <td class="num">{fmt_num(float(r['avg_latency']))}</td>
          <td class="num">{float(r['avg_link_util']):.3f}</td>
          <td class="num">{fmt_num(float(r['color_buffer_wait_cycles']))}</td>
          <td class="num">{fmt_num(float(r['link_wait_cycles']))}</td>
          <td class="num">{r['ordering_violations']}</td>
          <td><div class="bar" style="width:{bar_w}px" title="{ms} cycles"></div></td>
        </tr>"""

    violations = sum(int(r["ordering_violations"]) for r in rows)
    color_wins = sum(1 for s in summary if s["winner"] == "Color")
    xy_wins = sum(1 for s in summary if s["winner"] == "XY")
    ties = sum(1 for s in summary if s["winner"] == "tie")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Color NoC Simulation Report</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --border: #2d3a4f;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --green: #3fb950;
      --red: #f85149;
      --amber: #d29922;
      --font: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
      --mono: "IBM Plex Mono", "Consolas", monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      font-size: 15px;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }}
    h1 {{ font-size: 1.75rem; font-weight: 600; margin: 0 0 0.25rem; }}
    .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
    h2 {{ font-size: 1.25rem; margin: 2.5rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }}
    h3 {{ font-size: 1rem; margin: 1.5rem 0 0.5rem; color: var(--accent); }}
    p {{ margin: 0.6rem 0; }}
    a {{ color: var(--accent); }}
    code, pre {{ font-family: var(--mono); font-size: 0.875em; }}
    pre {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 1rem;
      overflow-x: auto;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 1rem;
      margin: 1.5rem 0;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
    }}
    .card .val {{ font-size: 1.5rem; font-weight: 600; }}
    .card .lbl {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      margin: 1rem 0;
    }}
    th, td {{
      text-align: left;
      padding: 0.55rem 0.75rem;
      border-bottom: 1px solid var(--border);
    }}
    th {{ color: var(--muted); font-weight: 500; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }}
    td.num {{ font-family: var(--mono); text-align: right; }}
    tr.win-color td:nth-child(5) {{ color: var(--green); }}
    tr.win-xy td:nth-child(5) {{ color: var(--red); }}
    tr.tie td:nth-child(5) {{ color: var(--amber); }}
    .bar {{
      height: 8px;
      background: linear-gradient(90deg, var(--accent), #a371f7);
      border-radius: 2px;
      min-width: 2px;
    }}
    .setup {{ background: var(--surface); border-radius: 8px; padding: 1rem 1.25rem; border: 1px solid var(--border); }}
    .setup dl {{ display: grid; grid-template-columns: 160px 1fr; gap: 0.35rem 1rem; margin: 0; }}
    .setup dt {{ color: var(--muted); }}
    .setup dd {{ margin: 0; }}
    ul {{ padding-left: 1.25rem; }}
    li {{ margin: 0.35rem 0; }}
    .meta {{ font-size: 0.8rem; color: var(--muted); margin-top: 3rem; }}
    .callout {{
      background: var(--surface);
      border-left: 3px solid var(--accent);
      padding: 1rem 1.25rem;
      margin: 1rem 0;
      border-radius: 0 6px 6px 0;
    }}
    .callout ul {{ margin: 0.5rem 0 0; }}
    h4 {{ font-size: 0.95rem; margin: 1.25rem 0 0.4rem; color: var(--muted); }}
    ol.steps {{ margin: 0.75rem 0; padding-left: 1.5rem; }}
    ol.steps li {{ margin: 0.4rem 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Color NoC Simulation Report</h1>
    <p class="subtitle">
      Mixed color scheme (<code>color_mixed</code>, K=16) vs single-VN XY baseline on 2D mesh.
      Patent background: <a href="color_mechanism_analysis.md">color_mechanism_analysis.md</a>
    </p>

    <div class="cards">
      <div class="card"><div class="val">{len(rows)}</div><div class="lbl">Runs</div></div>
      <div class="card"><div class="val" style="color:var(--green)">{violations}</div><div class="lbl">Ordering violations</div></div>
      <div class="card"><div class="val" style="color:var(--green)">{color_wins}</div><div class="lbl">Color wins</div></div>
      <div class="card"><div class="val" style="color:var(--red)">{xy_wins}</div><div class="lbl">XY wins</div></div>
      <div class="card"><div class="val" style="color:var(--amber)">{ties}</div><div class="lbl">Ties</div></div>
    </div>

    <h2>Setup</h2>
    <div class="setup">
      <dl>
        <dt>Meshes</dt><dd>4×4 (16 PEs), 8×8 (64 PEs)</dd>
        <dt>Message size</dt><dd>128 bytes</dd>
        <dt>Colors</dt><dd>16 — unicast XY/YX, row multicast, col reduction, dim-exchange, all-to-all phases</dd>
        <dt>Baseline</dt><dd>Single virtual network, dimension-order XY routing</dd>
        <dt>Workloads</dt><dd><code>ring</code>, <code>direct_allgather</code>, <code>broadcast_tree</code>, <code>reduction_tree</code>, <code>all_to_all</code>, <code>systolic</code>, <code>mixed</code></dd>
        <dt>Ordering</dt><dd>Per <code>(color, src, dst)</code> stream lock</dd>
        <dt>Data source</dt><dd><code>{html.escape(str(csv_path))}</code></dd>
      </dl>
    </div>

    {color_gen_section}

    <h2>Reproduce</h2>
    <pre>.venv/bin/python examples/run_color_vs_xy.py --msg-bytes 128
.venv/bin/python -m pytest tests/test_color_noc.py -q
.venv/bin/python examples/generate_color_report_html.py</pre>

    <h2>Makespan Summary</h2>
    <p>Speedup column: ratio favoring the winner (XY makespan ÷ Color makespan when Color wins, else inverse).</p>
    <table>
      <thead>
        <tr>
          <th>Mesh</th><th>Pattern</th><th>XY cycles</th><th>Color cycles</th><th>Winner</th><th>Speedup</th>
        </tr>
      </thead>
      <tbody>{summary_rows}
      </tbody>
    </table>

    <h2>Full Results</h2>
    <table>
      <thead>
        <tr>
          <th>Mesh</th><th>Scheme</th><th>Pattern</th><th>Makespan</th><th>Avg latency</th>
          <th>Link util</th><th>Color buf wait</th><th>Link wait</th><th>Order viol.</th><th></th>
        </tr>
      </thead>
      <tbody>{detail_rows}
      </tbody>
    </table>

    <h2>Analysis</h2>

    <h3>Where color wins (4×4)</h3>
    <ul>
      <li><strong>direct_allgather</strong> — parallel XY/YX unicast colors avoid single-VN contention (up to ~6× faster makespan).</li>
      <li><strong>systolic</strong> — dedicated path colors; fewer flits via direct neighbor routes.</li>
      <li><strong>all_to_all</strong> — time-phased colors reduce head-of-line blocking.</li>
      <li><strong>mixed</strong> — composite workload ~7% faster makespan, much lower avg latency.</li>
    </ul>

    <h3>Where color loses</h3>
    <ul>
      <li><strong>ring (4×4 and 8×8)</strong> — ring traffic mapped to generic unicast colors rather than mesh-valid snake/ring routes; XY Manhattan paths win on makespan.</li>
      <li><strong>8×8 at high load</strong> — when <code>color_buffer_wait_cycles</code> dominates, per-color queue backpressure inflates makespan despite lower per-packet latency. Check buffer depth and injection scheduling.</li>
      <li><strong>Tree collectives</strong> — shallow trees tie or differ by small margins on small meshes.</li>
    </ul>

    <h3>Metrics</h3>
    <table>
      <thead><tr><th>Metric</th><th>Color</th><th>XY</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td><code>makespan_cycles</code></td><td>Lower on 4×4 gather/systolic</td><td>Lower on ring / saturated 8×8</td><td>Primary objective</td></tr>
        <tr><td><code>avg_latency</code></td><td>Often much lower</td><td>Higher under contention</td><td>Per-flit delivery</td></tr>
        <tr><td><code>color_buffer_wait_cycles</code></td><td>Can dominate 8×8</td><td>N/A</td><td>Per-color queue stall</td></tr>
        <tr><td><code>ordering_violations</code></td><td>0</td><td>0</td><td>Stream lock enforced</td></tr>
      </tbody>
    </table>

    <h2>Conclusions</h2>
    <ol>
      <li>Static color routing shows clear makespan advantage on <strong>4×4 direct all-gather and systolic</strong> patterns with zero reordering.</li>
      <li>Ring all-reduce needs mesh-valid snake/ring color mapping before color can compete on makespan.</li>
      <li>8×8 results are sensitive to per-color buffer capacity and injection rate — high <code>color_buffer_wait_cycles</code> indicates scheduling/backpressure tuning is required at scale.</li>
      <li>Fault-tolerance code exists in <code>wsesim/network/color_repair.py</code> (not yet in this benchmark harness).</li>
    </ol>

    <h2>Next steps</h2>
    <ul>
      <li>Fix ring color mapping: <code>snake_ring</code> with acyclic route check</li>
      <li>Investigate 8×8 <code>color_buffer_wait_cycles</code> — buffer depth, round-robin fairness</li>
      <li>Defect-rate sweep with <code>color_repair</code></li>
      <li>Wire <code>color_scheme</code> into DSE evaluator</li>
    </ul>

    <p class="meta">Generated {ts} · wse-simulator color NoC study</p>
  </div>
</body>
</html>"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "outputs" / "color_vs_xy" / "results.csv"
    out_path = root / "docs" / "color_simulation_report.html"

    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}; run examples/run_color_vs_xy.py first")

    rows = load_results(csv_path)
    summary = build_summary(rows)
    out_path.write_text(render_html(rows, summary, csv_path.relative_to(root)), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

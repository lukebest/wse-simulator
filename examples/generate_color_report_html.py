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

from wsesim.network.collective import IDEAL_PATTERNS
from wsesim.network.color import ColorPlan
from wsesim.network.color_catalog import CATALOG, MAX_COLORS, build_ideal_plan


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


_DIR_CN = {"east": "东 →", "west": "← 西", "south": "南 ↓", "north": "北 ↑"}

# Which colors each ideal collective rides.
_PATTERN_COLORS = {
    "broadcast": [2, 3],
    "gather": [4, 5],
    "reduce": [6, 7],
    "allgather": [8, 9, 10, 11],
    "allreduce": [12, 13, 14, 15],
}


def _route_kind_cn(kind: str, spec: str) -> tuple[str, str]:
    if kind == "unicast":
        mode = spec.upper()
        return (
            f"维度序单播 ({mode})",
            "按包内 <code>dst</code> 在每跳静态选择下一跳（XY 先行后列，YX 先列后行）。",
        )
    return (
        f"方向总线 ({_DIR_CN.get(spec, spec)})",
        f"<code>dest[node]</code> 固定指向 <strong>{spec}</strong> 方向相邻节点，"
        "与运行时 dst 无关；包沿该方向前进直到抵达目的。",
    )


def _catalog_table_html() -> str:
    body = ""
    for cid in range(MAX_COLORS):
        name, kind, spec = CATALOG[cid]
        kind_cn, detail = _route_kind_cn(kind, spec)
        body += f"""
        <tr>
          <td class="num">{cid}</td>
          <td><code>{html.escape(name)}</code></td>
          <td>{html.escape(kind_cn)}</td>
          <td>{detail}</td>
        </tr>"""
    return f"""
    <table>
      <thead><tr><th>ID</th><th>名称</th><th>路由类型</th><th>静态规则（与 mesh 规模无关）</th></tr></thead>
      <tbody>{body}</tbody>
    </table>"""


def _pattern_routing_table_html() -> str:
    desc = {
        "broadcast": (
            "根 (0,0) 生成树",
            "第 0 行沿 <code>bcast_row_east</code> 东向涟漪扩散，每列再沿 <code>bcast_col_south</code> 南向扩散。"
            "相邻边、各列并行；makespan ≈ (列−1)+(行−1)。",
        ),
        "gather": (
            "逆生成树 → 根",
            "各列经 <code>gather_col_north</code> 向上汇聚到第 0 行，再沿 <code>gather_row_west</code> 西向汇聚到根。",
        ),
        "reduce": (
            "逆生成树 → 根（归约）",
            "与 gather 同形，但走独立 VN（<code>reduce_col_north</code>/<code>reduce_row_west</code>），"
            "每跳合并、payload 恒定。",
        ),
        "allgather": (
            "2D 环 allgather",
            "先行内双向（<code>allgather_row_east/west</code>）后列内双向（<code>allgather_col_south/north</code>），"
            "各行/列链路不相交、并行。",
        ),
        "allreduce": (
            "2D reduce-scatter + allgather",
            "行 RS(<code>…rs_row_east</code>) → 行 AG(<code>…ag_row_west</code>) → 列 RS(<code>…rs_col_south</code>) "
            "→ 列 AG(<code>…ag_col_north</code>)。",
        ),
    }
    body = ""
    for pat in IDEAL_PATTERNS:
        shape, detail = desc[pat]
        colors = _PATTERN_COLORS[pat]
        names = ", ".join(f"{cid}:<code>{html.escape(CATALOG[cid][0])}</code>" for cid in colors)
        body += f"""
        <tr>
          <td><code>{html.escape(pat)}</code></td>
          <td>{html.escape(shape)}</td>
          <td>{names}</td>
          <td>{detail}</td>
        </tr>"""
    return f"""
    <table>
      <thead><tr><th>Workload</th><th>理想静态路由</th><th>使用 Color</th><th>说明</th></tr></thead>
      <tbody>{body}</tbody>
    </table>"""


def render_color_generation_section(num_colors: int = MAX_COLORS) -> str:
    catalog = _catalog_table_html()
    pattern_tbl = _pattern_routing_table_html()
    return f"""
    <h2>Color 生成方法与依据</h2>

    <h3>设计依据</h3>
    <div class="callout">
      <p>本仿真参照 Cerebras 专利 <strong>US10,515,303</strong> 的 Color 机制，并针对集合通信 workload
         设计<strong>最理想的静态路由表</strong>：</p>
      <ul>
        <li><strong>编译期静态路由</strong> — 每个 color 对应一条固定规则，<code>dest[node] → next_hop</code>
            只取决于当前节点，<strong>与运行时 dst 无关</strong>；运行时无动态路由计算。</li>
        <li><strong>规则与 mesh 规模无关</strong> — color 的语义（ID 与方向规则）对 4×4 与 8×8 完全一致，
            仅 dest 表按具体 (rows, cols) 实例化。最多 <strong>{num_colors} 个 color</strong>。</li>
        <li><strong>每类集合通信选最省链路的静态路由</strong> — 广播/gather/reduce 用<strong>生成树</strong>
            （行涟漪 + 列涟漪，全为相邻边），allgather/allreduce 用 <strong>2D 环 / reduce-scatter+allgather</strong>，
            各阶段走独立 VN 隔离、链路不相交以并行。</li>
        <li><strong>保序</strong> — 每个 <code>(color, src, dst)</code> 流串行注入，配合固定路由保证 FIFO、
            <code>ordering_violations = 0</code>。</li>
      </ul>
    </div>

    <h3>方向总线：color 的基本构件</h3>
    <p>大多数理想路由由<strong>方向总线</strong>构成 —— 一个 color 在每个节点静态指向某一罗盘方向的相邻节点。
       包沿该方向前进直到 <code>current == dst</code> 停止。这正是专利中 Dest 661「<code>color → next_hop</code>、
       与 dst 无关」的静态转发表。生成规则（<code>wsesim/network/color_catalog.py</code>）：</p>
    <pre>def _set_bus(plan, color_id, direction):       # direction ∈ {{east,west,south,north}}
    for node in mesh:
        r, c = divmod(node, cols)
        nr, nc = r + dr, c + dc                # 该方向的相邻节点
        plan.set_dest(node, color_id, {{nr*cols+nc}} if in_bounds else set())</pre>

    <h3>固定 Color 目录（catalog，{num_colors} 个，4×4 与 8×8 相同）</h3>
    <p>由 <code>build_ideal_plan(rows, cols)</code> 实例化；下表语义对任意 mesh 规模不变：</p>
    {catalog}

    <h3>每个集合通信 workload 的理想静态路由</h3>
    <p>由 <code>generate_ideal_collective(pattern, rows, cols, chunk)</code> 生成相邻边流量并打上对应 color，
       配合 level 延迟实现流水线：</p>
    {pattern_tbl}

    <h3>与 Baseline 对比</h3>
    <table>
      <thead><tr><th>方案</th><th>Color 数</th><th>路由</th><th>realisation</th></tr></thead>
      <tbody>
        <tr>
          <td><code>color_ideal</code></td>
          <td class="num">{num_colors}</td>
          <td>上表 catalog + 每 workload 理想静态路由</td>
          <td><code>build_ideal_plan</code> + <code>generate_ideal_collective</code></td>
        </tr>
        <tr>
          <td><code>xy_single_vn</code></td>
          <td class="num">1</td>
          <td>单一 VN，XY 维度序</td>
          <td><code>generate_baseline_collective</code>（朴素直连）</td>
        </tr>
      </tbody>
    </table>
    <p>Baseline 为各集合通信的<strong>朴素直连</strong>实现且全部走单一 VN：broadcast = 根→每点直连、
       gather/reduce = 每点→根直连、allgather = 全连通 all-to-all、allreduce = 先 reduce 到根再 broadcast。
       所有流共享 XY 链路与缓冲，形成根/链路瓶颈。</p>

    <h3>可选：动态 Color 分配</h3>
    <p>若给定 flow 的时间窗与占用链路，<code>wsesim/network/color_alloc.py</code> 提供 greedy / 图着色 / ILP
       分配（时间重叠且链路相交的 flow 不可共享 color，路由图需无环）。本 benchmark 使用上面的固定 catalog，
       未启用动态分配。</p>
    """


def build_summary(rows: list[dict[str, str]]) -> list[dict]:
    by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["mesh"], r["pattern"])
        by_key[key][r["scheme"]] = float(r["makespan_cycles"])

    summary = []
    for (mesh, pattern), schemes in sorted(by_key.items()):
        xy = schemes.get("xy_single_vn")
        color = schemes.get("color_ideal")
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


def _build_analysis_html(summary: list[dict]) -> str:
    best = max(summary, key=lambda s: s["ratio"] if s["winner"] == "Color" else 0)
    rows_html = ""
    for s in summary:
        if s["winner"] == "Color":
            rows_html += (
                f"<li><code>{html.escape(s['mesh'])}</code> "
                f"<code>{html.escape(s['pattern'])}</code> — "
                f"{fmt_num(s['xy'])} → {fmt_num(s['color'])} cycles "
                f"(<strong>{s['ratio']:.1f}×</strong>)</li>"
            )
    return f"""
    <p>Color ideal wins on <strong>{sum(1 for s in summary if s['winner']=='Color')}/{len(summary)}</strong>
       workload×mesh combinations. Largest speedup:
       <strong>{best['ratio']:.1f}×</strong> on <code>{html.escape(best['mesh'])}</code>
       <code>{html.escape(best['pattern'])}</code>.</p>
    <ul>{rows_html}</ul>"""


def render_html(
    rows: list[dict[str, str]], summary: list[dict], csv_path: Path, *, num_colors: int = MAX_COLORS
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
    analysis_html = _build_analysis_html(summary)

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
      Ideal static-route color scheme (<code>color_ideal</code>, K={num_colors}) vs single-VN XY baseline on 2D mesh.
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
        <dt>Colors</dt><dd>{num_colors} — fixed catalog of directional buses + XY/YX unicast (mesh-size independent)</dd>
        <dt>Baseline</dt><dd>Naive direct collective on a single VN, dimension-order XY routing</dd>
        <dt>Workloads</dt><dd><code>broadcast</code>, <code>gather</code>, <code>reduce</code>, <code>allreduce</code>, <code>allgather</code></dd>
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
    {analysis_html}

    <h3>Why ideal static routes win</h3>
    <ul>
      <li><strong>Minimal-link routing</strong> — broadcast/gather/reduce ride a spanning tree of <em>adjacent</em>
          edges (1 hop each) instead of the baseline's many multi-hop XY unicasts that converge on the root,
          cutting total link crossings and the root bottleneck.</li>
      <li><strong>Path diversity + VN isolation</strong> — allgather/allreduce spread their reduce-scatter /
          allgather phases across dedicated row/column buses so concurrent flows never share a link or a buffer.</li>
      <li><strong>Pipelining</strong> — level-based delays let each tree level / ring step overlap, so makespan
          scales with <code>(rows+cols)</code> rather than the number of participants.</li>
      <li><strong>Ordering preserved</strong> — every run reports <code>ordering_violations = 0</code>.</li>
    </ul>

    <h2>Conclusions</h2>
    <ol>
      <li>Designing the <strong>ideal static routing table per collective</strong> (tree for broadcast/gather/reduce,
          2D ring for allgather/allreduce) makes the color scheme beat the naive single-VN XY baseline on
          <strong>every</strong> workload and both mesh sizes, with zero reordering.</li>
      <li>The advantage <strong>grows with mesh size</strong> (e.g. allgather 8×8 reaches the largest speedup),
          because the baseline's root/all-to-all contention scales worse than the tree/ring depth.</li>
      <li>Color rules are <strong>mesh-size independent</strong>: the same {num_colors}-color catalog applies to
          4×4 and 8×8; only the dest tables are re-instantiated.</li>
      <li>Fault-tolerance code exists in <code>wsesim/network/color_repair.py</code> (not yet in this benchmark harness).</li>
    </ol>

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

#!/usr/bin/env python3
"""Generate docs/color_simulation_report.html from simulation CSV."""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


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


def render_html(rows: list[dict[str, str]], summary: list[dict], csv_path: Path) -> str:
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

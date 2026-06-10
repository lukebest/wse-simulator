#!/usr/bin/env python3
"""Generate TDM FB color sub-topology HTML for rectangular meshes via
**restriction** of a larger (k=2, n=…) hypercube FB.

Strategy: take parent 8×8 (2,6) or 16×16 (2,8), keep only logical links
whose endpoints + entire XY physical path stay inside the target sub-rect,
reuse parent color assignment. This trades logical link density and full
FB regularity for a much smaller Color count.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from wsesim.network.tdm_coloring import assign_colors
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly


PALETTE_BASE = [
    "#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626",
    "#0891b2", "#ca8a04", "#db2777", "#4f46e5", "#059669",
    "#7c3aed", "#c2410c", "#0d9488", "#be123c", "#1d4ed8", "#854d0e",
]


def restrict(parent_rows, parent_cols, k, n, keep_rows, keep_cols):
    topo = TDMFlatButterfly(k=k, n=n, rows=parent_rows, cols=parent_cols)
    parent_plan = topo.coloring()

    def in_sub(node):
        r, c = divmod(node, parent_cols)
        return r < keep_rows and c < keep_cols

    def path_ok(path):
        for u, v in path:
            if not (in_sub(u) and in_sub(v)):
                return False
        return True

    kept_links = []
    kept_paths = {}
    for u, v, dim in topo.logical_links():
        if not (in_sub(u) and in_sub(v)):
            continue
        path = topo.physical_path(u, v)
        if not path_ok(path):
            continue
        kept_links.append((u, v, dim))
        kept_paths[(u, v)] = path

    sub_phys = []
    for r in range(keep_rows):
        for c in range(keep_cols):
            nid = r * parent_cols + c
            if c + 1 < keep_cols:
                right = r * parent_cols + c + 1
                sub_phys.append((nid, right))
                sub_phys.append((right, nid))
            if r + 1 < keep_rows:
                down = (r + 1) * parent_cols + c
                sub_phys.append((nid, down))
                sub_phys.append((down, nid))
    sub_plan = assign_colors(kept_links, kept_paths, sub_phys)
    return topo, parent_plan, kept_links, kept_paths, sub_plan


def restricted_to_renumbered(node, parent_cols, keep_cols):
    r, c = divmod(node, parent_cols)
    return r * keep_cols + c


def degree_histogram(kept_links, keep_rows, keep_cols, parent_cols):
    deg = Counter()
    for u, v, _ in kept_links:
        deg[u] += 1
    full = [deg.get(r * parent_cols + c, 0) for r in range(keep_rows) for c in range(keep_cols)]
    return Counter(full), full


def missing_dim_per_node(kept_links, keep_rows, keep_cols, parent_cols, n_dims):
    """For each node, which dims still have a logical partner inside the sub-mesh."""
    have = {(r * parent_cols + c): set() for r in range(keep_rows) for c in range(keep_cols)}
    for u, _, dim in kept_links:
        have[u].add(dim)
    miss_count = Counter()
    miss_per_node = {}
    for n_id, dims in have.items():
        missing = sorted(set(range(n_dims)) - dims)
        miss_per_node[n_id] = missing
        miss_count[len(missing)] += 1
    return miss_count, miss_per_node


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{ --bg:#0f1419; --panel:#1a2332; --text:#e8edf4; --muted:#94a3b8; --border:#2d3a4f; --mesh:#334155; --accent:#fbbf24; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
  header.hero {{ padding:2rem; background:linear-gradient(135deg,#1e293b,#0f172a); border-bottom:1px solid var(--border); }}
  header.hero h1 {{ margin:0 0 .5rem; font-size:1.6rem; }}
  header.hero .lead {{ color:var(--accent); font-weight:600; font-size:.95rem; margin:.3rem 0 .5rem; }}
  header.hero p {{ margin:.25rem 0; color:var(--muted); max-width:62rem; }}
  main {{ max-width:1400px; margin:0 auto; padding:1.75rem 1.5rem 3rem; }}
  section {{ margin-bottom:2rem; }}
  h2 {{ font-size:1.2rem; margin:0 0 .75rem; }}
  h3 {{ font-size:1rem; margin:1rem 0 .5rem; color:#cbd5e1; }}
  .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.75rem; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:.85rem 1rem; }}
  .stat .val {{ font-size:1.5rem; font-weight:700; }}
  .stat .val.win {{ color:#86efac; }}
  .stat .val.cost {{ color:#fda4af; }}
  .stat .lbl {{ color:var(--muted); font-size:.8rem; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:1rem 1.25rem; margin-bottom:1.25rem; }}
  .compare-table th {{ background:#1e3a8a; }}
  .compare-table td.win {{ color:#86efac; }}
  .compare-table td.cost {{ color:#fda4af; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
  th,td {{ border:1px solid var(--border); padding:.4rem .55rem; text-align:left; }}
  th {{ background:#243044; }}
  .svg-wrap {{ background:#111827; border-radius:8px; padding:.35rem; overflow:auto; }}
  .topo-svg {{ width:100%; max-width:{svg_max}px; height:auto; display:block; margin:0 auto; }}
  .mesh-edge {{ stroke:var(--mesh); stroke-width:1; opacity:.25; pointer-events:none; }}
  .logic-link {{ fill:none; stroke-width:1.5; opacity:.78; cursor:pointer; pointer-events:stroke; stroke-linecap:round; transition:opacity .12s,stroke-width .12s; }}
  .logic-link:hover,.logic-link.hovered {{ opacity:1; stroke-width:2.8; }}
  .logic-link.dimmed {{ opacity:.06; }}
  .node {{ fill:#1e293b; stroke:#64748b; stroke-width:1; pointer-events:none; }}
  .node.partial {{ stroke:#fbbf24; stroke-width:2; }}
  .node-label {{ fill:#e2e8f0; font-size:{node_font}px; font-family:ui-monospace,monospace; text-anchor:middle; pointer-events:none; }}
  .filter-bar {{ display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:.5rem; }}
  .filter-btn {{ border:1px solid var(--border); background:#243044; color:var(--text); padding:.3rem .55rem; border-radius:5px; font-size:.78rem; cursor:pointer; display:flex; align-items:center; gap:.3rem; font-family:ui-monospace,monospace; }}
  .filter-btn.active {{ background:#334155; }}
  .filter-btn.inactive {{ opacity:.35; }}
  .filter-actions {{ display:flex; gap:.5rem; margin-bottom:.5rem; }}
  .filter-actions button {{ background:#1e3a8a; border:1px solid #3b82f6; color:#dbeafe; padding:.3rem .65rem; border-radius:5px; cursor:pointer; font-size:.78rem; }}
  .filter-actions button.alt {{ background:#374151; border-color:#6b7280; color:#e5e7eb; }}
  .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
  .tooltip {{ position:fixed; z-index:999; pointer-events:none; background:#0f172a; border:1px solid #475569; border-radius:5px; padding:.35rem .55rem; font-family:ui-monospace,monospace; font-size:.78rem; display:none; box-shadow:0 4px 12px rgba(0,0,0,.4); white-space:nowrap; }}
  .tooltip .tag {{ color:#94a3b8; margin-left:.3rem; font-size:.72rem; }}
  .hover-info {{ margin-top:.5rem; font-family:ui-monospace,monospace; font-size:.8rem; color:#cbd5e1; min-height:1.2em; }}
  .color-viewer select {{ background:#243044; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:.4rem .6rem; font-size:.85rem; }}
  code {{ background:#243044; padding:.1rem .35rem; border-radius:3px; font-size:.85em; }}
  .legend-row {{ display:flex; gap:1.5rem; flex-wrap:wrap; font-size:.78rem; color:var(--muted); margin-top:.5rem; }}
  .legend-row span.swatch {{ display:inline-block; width:14px; height:14px; border-radius:50%; vertical-align:middle; margin-right:.3rem; }}
  .legend-row span.swatch.partial {{ background:#1e293b; border:2px solid #fbbf24; }}
  .legend-row span.swatch.full {{ background:#1e293b; border:1px solid #64748b; }}
  details summary {{ cursor:pointer; padding:.4rem 0; font-weight:600; }}
  .rule-box {{ font-family:ui-monospace,monospace; background:#111827; border-radius:6px; padding:.75rem 1rem; font-size:.8rem; margin:.5rem 0; line-height:1.6; }}
  footer {{ text-align:center; color:var(--muted); font-size:.75rem; padding:1.5rem; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<header class="hero">
  <h1>{title}</h1>
  <p class="lead">从父 mesh <code>{parent_rows}×{parent_cols}</code> 的 <code>(k=2, n={n})</code> 超立方 FB 限制而来，C 从 {full_C} → <strong>{C}</strong>（满 FB 需 C={full_FB_C}）</p>
  <p>{intro}</p>
</header>
<main>
  <section>
    <h2>核心指标</h2>
    <div class="summary-grid">
      <div class="stat"><div class="val win">{C}</div><div class="lbl">TDM Color 数</div></div>
      <div class="stat"><div class="val">{n_links}</div><div class="lbl">有向逻辑链路</div></div>
      <div class="stat"><div class="val cost">{deg_min}–{deg_max}</div><div class="lbl">节点度（满 hypercube={n}）</div></div>
      <div class="stat"><div class="val">{n_pe}</div><div class="lbl">PE 数</div></div>
      <div class="stat"><div class="val">{partial_nodes}</div><div class="lbl">退化节点数（缺 ≥1 维）</div></div>
    </div>
  </section>

  <section class="panel">
    <h2>与全 FB 切分方案对比</h2>
    <table class="compare-table">
      <thead><tr><th>方案</th><th>逻辑链路</th><th>Color 数 C</th><th>节点度</th><th>逻辑直径</th><th>规则性</th></tr></thead>
      <tbody>
        <tr>
          <td>矩形 2-flat 全 FB（K_cols + K_rows）</td>
          <td>{full_FB_links}</td>
          <td class="cost">{full_FB_C}</td>
          <td>{full_FB_degree}</td>
          <td>2</td>
          <td>完全对称</td>
        </tr>
        <tr>
          <td><strong>本方案：限制版超立方 FB</strong></td>
          <td class="cost">{n_links}</td>
          <td class="win"><strong>{C}</strong></td>
          <td class="cost">{deg_min}–{deg_max}</td>
          <td class="cost">≤ {logical_diam}</td>
          <td class="cost">不对称（{partial_nodes} 节点退化）</td>
        </tr>
      </tbody>
    </table>
    <h3>切分规则</h3>
    <div class="rule-box">{rule_text}</div>
  </section>

  <section class="panel">
    <h2>{C} Color 合并预览</h2>
    <div class="filter-actions">
      <button id="select-all">全选</button>
      <button id="deselect-all" class="alt">全不选</button>
    </div>
    <div class="filter-bar" id="filters"></div>
    <div class="svg-wrap" id="merged"></div>
    <div class="legend-row">
      <span><span class="swatch full"></span>满度节点（{n} 个逻辑邻居）</span>
      <span><span class="swatch partial"></span>退化节点（缺 ≥1 维邻居）</span>
    </div>
    <div class="hover-info" id="hover-info">将鼠标移到链路上…</div>
  </section>

  <section class="panel color-viewer">
    <h2>单 Color 子拓扑</h2>
    <label>选择 Color：<select id="single-color">{select_options}</select></label>
    <div class="svg-wrap" id="single" style="margin-top:.75rem"></div>
  </section>

  <section class="panel">
    <h2>Color 链路分配</h2>
    <table><thead><tr><th>Color</th><th></th><th>链路数</th><th>含义</th></tr></thead><tbody>{color_table_rows}</tbody></table>
  </section>

  <section class="panel">
    <h2>物理跳数分布</h2>
    <table><thead><tr><th>物理跳数</th><th>链路数</th><th>说明</th></tr></thead><tbody>{phys_rows}</tbody></table>
    <p style="color:var(--muted);font-size:.83rem;margin-top:.5rem">
      物理跳数只能为 1, 2, 4, 8, …（k=2 维度翻转 = 行/列偏移 2 的幂），且受限于子 mesh 尺寸。
    </p>
  </section>

  <section class="panel">
    <h2>节点度数分布</h2>
    <table><thead><tr><th>度数</th><th>节点数</th><th>占比</th></tr></thead><tbody>{deg_rows}</tbody></table>
    <h3>缺失维度统计（按缺失数）</h3>
    <table><thead><tr><th>缺失维度数</th><th>节点数</th></tr></thead><tbody>{miss_rows}</tbody></table>
  </section>
</main>
<footer>{footer_path} · 生成自 scripts/generate_tdm_fb_rect_restricted_docs.py</footer>

<script>
const LINKS = {links_json};
const COLORS = {colors_json};
const PARTIAL_NODES = new Set({partial_nodes_json});
const ROWS={rows}, COLS={cols}, CELL={cell}, MARGIN={margin}, W={W}, H={H}, NRADIUS={nradius};

function nodeXY(n) {{ const r=Math.floor(n/COLS), c=n%COLS; return [MARGIN+c*CELL, MARGIN+r*CELL]; }}

function arcPath(src,dst,dim,idx) {{
  const [x0,y0]=nodeXY(src), [x1,y1]=nodeXY(dst);
  const sc=src%COLS, dc=dst%COLS, sr=Math.floor(src/COLS), dr=Math.floor(dst/COLS);
  const dh = Math.abs(dc-sc), dv = Math.abs(dr-sr);
  let cx,cy;
  if (dh >= dv) {{ const s=src<dst?-1:1; cx=(x0+x1)/2; cy=(y0+y1)/2+s*(10+dh*5)*(idx%2?0.7:1); }}
  else {{ const s=src<dst?-1:1; cx=(x0+x1)/2+s*(10+dv*5)*(idx%2?0.7:1); cy=(y0+y1)/2; }}
  return `M ${{x0}} ${{y0}} Q ${{cx}} ${{cy}} ${{x1}} ${{y1}}`;
}}

function markers(ids) {{
  let s='<defs>'; for (const id of ids) s+=`<marker id="a${{id}}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="${{COLORS[id].hex}}"/></marker>`;
  return s+'</defs>';
}}
function nodes() {{
  let s='';
  for(let n=0;n<ROWS*COLS;n++){{
    const[x,y]=nodeXY(n);
    const cls = PARTIAL_NODES.has(n) ? 'node partial' : 'node';
    s+=`<circle cx="${{x}}" cy="${{y}}" r="${{NRADIUS}}" class="${{cls}}"/><text x="${{x}}" y="${{y+3}}" class="node-label">${{n}}</text>`;
  }}
  return s;
}}
function skeleton() {{
  let s='';
  for(let n=0;n<ROWS*COLS;n++){{
    const[x0,y0]=nodeXY(n); const r=Math.floor(n/COLS),c=n%COLS;
    if(c+1<COLS){{const[x1,y1]=nodeXY(n+1); s+=`<line x1="${{x0}}" y1="${{y0}}" x2="${{x1}}" y2="${{y1}}" class="mesh-edge"/>`;}}
    if(r+1<ROWS){{const[x1,y1]=nodeXY(n+COLS); s+=`<line x1="${{x0}}" y1="${{y0}}" x2="${{x1}}" y2="${{y1}}" class="mesh-edge"/>`;}}
  }}
  return s;
}}

function render(container, activeSet, infoEl) {{
  let svg=`<svg viewBox="0 0 ${{W}} ${{H}}" class="topo-svg">${{markers([...activeSet])}}${{nodes()}}${{skeleton()}}`;
  const cnt={{}}; activeSet.forEach(c=>cnt[c]=0);
  for (const l of LINKS) {{
    if (!activeSet.has(l.color)) continue;
    const i=cnt[l.color]++; const d=arcPath(l.src,l.dst,l.dim,i);
    const label=`${{l.src}}→${{l.dst}} · dim${{l.dim}} · C${{l.color}} · phys${{l.phys}}`;
    svg+=`<path d="${{d}}" class="logic-link" stroke="${{COLORS[l.color].hex}}" marker-end="url(#a${{l.color}})" data-label="${{label}}"/>`;
  }}
  svg+='</svg>'; container.innerHTML=svg;
  attachHover(container.querySelector('svg'), infoEl);
}}

const tip=document.getElementById('tooltip');
function attachHover(svg, infoEl) {{
  if(!svg) return;
  const all=svg.querySelectorAll('.logic-link');
  all.forEach(el=>{{
    el.onmouseenter=e=>{{ const lb=el.dataset.label; const p=lb.split(' · '); tip.innerHTML=`${{p[0]}} <span class="tag">${{p.slice(1).join(' · ')}}</span>`; tip.style.display='block'; tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY-8)+'px'; all.forEach(x=>x.classList.toggle('dimmed',x!==el)); el.classList.add('hovered'); if(infoEl)infoEl.textContent=lb; }};
    el.onmousemove=e=>{{ tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY-8)+'px'; }};
    el.onmouseleave=()=>{{ tip.style.display='none'; all.forEach(x=>x.classList.remove('dimmed','hovered')); if(infoEl)infoEl.textContent='将鼠标移到链路上…'; }};
  }});
}}

const merged=document.getElementById('merged');
const filters=document.getElementById('filters');
let active=new Set(COLORS.map(c=>c.id));
const btnMap=new Map();
COLORS.forEach(c=>{{
  const b=document.createElement('button');
  b.className='filter-btn active';
  b.innerHTML=`<span class="dot" style="background:${{c.hex}}"></span>C${{c.id}}`;
  b.onclick=()=>{{
    if(active.has(c.id)){{ active.delete(c.id); b.classList.replace('active','inactive'); }}
    else{{ active.add(c.id); b.classList.replace('inactive','active'); }}
    render(merged,active,document.getElementById('hover-info'));
  }};
  filters.appendChild(b);
  btnMap.set(c.id,b);
}});
function refreshButtons() {{
  COLORS.forEach(c=>{{
    const b=btnMap.get(c.id);
    if (active.has(c.id)) b.className='filter-btn active'; else b.className='filter-btn inactive';
  }});
}}
document.getElementById('select-all').onclick=()=>{{ COLORS.forEach(c=>active.add(c.id)); refreshButtons(); render(merged,active,document.getElementById('hover-info')); }};
document.getElementById('deselect-all').onclick=()=>{{ active.clear(); refreshButtons(); render(merged,active,document.getElementById('hover-info')); }};
render(merged,active,document.getElementById('hover-info'));

const single=document.getElementById('single');
const sel=document.getElementById('single-color');
function renderSingle(){{ render(single, new Set([+sel.value]), null); }}
sel.onchange=renderSingle; renderSingle();
</script>
</body></html>
"""


CONFIGS = [
    {
        "parent_rows": 8, "parent_cols": 8, "k": 2, "n": 6,
        "keep_rows": 6, "keep_cols": 8,
        "full_FB_C": 16, "full_FB_links": 576, "full_FB_degree": "5+7=12",
        "intro": (
            "从 8×8 (2,6) 超立方 FB 中保留位于前 6 行内的所有逻辑链路，"
            "并丢弃任何 XY 物理路径经过被裁剪行（行 6,7）的链路。"
            "复用父着色 → C=5，全部 5 色仍被覆盖。"
        ),
        "rule": (
            "<b>父结构</b>：8×8 mesh 上 (k=2, n=6) FB；行 r=d₅+2d₃+4·(d₅⊕…)，列 c=d₀+2d₁+4d₂；"
            "每节点恰好 6 个逻辑邻居（每维 1 个 hypercube 翻转伙伴）。<br>"
            "<b>裁剪</b>：丢弃 d₅d₄=11 对应的行 6–7（16 节点），保留 48 节点 = 6×8 mesh。<br>"
            "<b>受影响的链路</b>：dim-5 (行 ±4 翻转) 中 row∈{{2,3}} → row{{6,7}} 共 32 条丢弃；"
            "dim-4 (行 ±2) 中 row∈{{4,5}}→{{6,7}} 共 32 条丢弃；其他维全部保留。<br>"
            "<b>结果</b>：256 条有向链路（满 FB 576 的 44%），C 从 16 → <b>5</b>。"
        ),
    },
    {
        "parent_rows": 16, "parent_cols": 16, "k": 2, "n": 8,
        "keep_rows": 12, "keep_cols": 16,
        "full_FB_C": 64, "full_FB_links": 4992, "full_FB_degree": "11+15=26",
        "intro": (
            "从 16×16 (2,8) 超立方 FB 中保留位于前 12 行内的所有逻辑链路，"
            "并丢弃任何物理路径经过行 12–15 的链路。复用父着色 → C=10。"
        ),
        "rule": (
            "<b>父结构</b>：16×16 mesh 上 (k=2, n=8) FB，每节点 8 个 hypercube 邻居；父 C=10。<br>"
            "<b>裁剪</b>：丢弃行 12–15（64 节点），保留 192 节点 = 12×16 mesh。<br>"
            "<b>受影响的链路</b>：dim-7 (行 ±8 翻转) 全部受 12 行约束；"
            "dim-5/6 (行 ±2, ±4) 中起点/终点在被丢弃行的链路丢弃。<br>"
            "<b>结果</b>：1 408 条有向链路（满 FB 4 992 的 28%），C 从 64 → <b>10</b>。"
        ),
    },
    {
        "parent_rows": 16, "parent_cols": 16, "k": 2, "n": 8,
        "keep_rows": 14, "keep_cols": 14,
        "full_FB_C": 49, "full_FB_links": 5096, "full_FB_degree": "13+13=26",
        "intro": (
            "从 16×16 (2,8) 超立方 FB 中保留位于左上 14×14 区域内的所有逻辑链路，"
            "并丢弃任何物理路径经过行 14–15 或列 14–15 的链路。复用父着色 → C=10。"
        ),
        "rule": (
            "<b>父结构</b>：16×16 mesh 上 (k=2, n=8) FB；行/列各 4 位地址；父 C=10。<br>"
            "<b>裁剪</b>：丢弃行 14–15 与列 14–15（共 60 节点），保留 196 节点 = 14×14 mesh。<br>"
            "<b>受影响的链路</b>：dim-3 (列 ±8)、dim-7 (行 ±8) 全部受边界约束；"
            "短距离 dim 也有部分链路终点落入被裁剪边角而丢弃。<br>"
            "<b>结果</b>：1 400 条有向链路（满 FB 5 096 的 27%），C 从 49 → <b>10</b>。"
        ),
    },
]


def write_html(cfg: dict) -> Path:
    pr, pc = cfg["parent_rows"], cfg["parent_cols"]
    kr, kc = cfg["keep_rows"], cfg["keep_cols"]
    k, n = cfg["k"], cfg["n"]
    topo, parent_plan, kept_links, kept_paths, sub_plan = restrict(pr, pc, k, n, kr, kc)
    C = sub_plan.C
    n_links = len(kept_links)

    by_color = Counter(sub_plan.color_of_logical.values())
    by_phys = Counter(len(p) for p in kept_paths.values())

    deg_hist, deg_per_node = degree_histogram(kept_links, kr, kc, pc)
    miss_count, miss_per_node = missing_dim_per_node(kept_links, kr, kc, pc, n)
    partial_nodes = sum(c for d, c in miss_count.items() if d > 0)
    partial_list = sorted(nid for nid, m in miss_per_node.items() if m)

    palette = PALETTE_BASE * ((C // len(PALETTE_BASE)) + 1)
    colors_meta = [{"id": c, "hex": palette[c]} for c in range(C)]

    # Renumber nodes from parent grid (parent_cols stride) to sub-mesh (keep_cols stride)
    def renum(node):
        r, c = divmod(node, pc)
        return r * kc + c

    enriched_links = []
    for u, v, dim in kept_links:
        col = sub_plan.color_of_logical[(u, v)]
        enriched_links.append({
            "src": renum(u), "dst": renum(v), "dim": dim,
            "color": col, "phys": len(kept_paths[(u, v)]),
        })
    partial_renum = sorted(renum(nid) for nid in partial_list)

    # Hop-distance semantic note (k=2 only allows powers of 2 with cap)
    hop_notes = {1: "邻接（1 维翻转）", 2: "次维翻转或 2 跳路径", 4: "高维翻转 4 跳", 8: "最高维翻转 8 跳"}
    phys_rows = "".join(
        f"<tr><td>{d}</td><td>{by_phys[d]}</td><td>{hop_notes.get(d, '')}</td></tr>"
        for d in sorted(by_phys)
    )

    color_dim_note = {}
    for c in range(C):
        dims = Counter()
        for u, v, dim in kept_links:
            if sub_plan.color_of_logical[(u, v)] == c:
                dims[dim] += 1
        if dims:
            color_dim_note[c] = ", ".join(f"dim{d}×{cnt}" for d, cnt in sorted(dims.items()))
        else:
            color_dim_note[c] = "—"
    color_table_rows = "".join(
        f"<tr><td>C{c}</td><td><span class='dot' style='background:{palette[c]}'></span></td><td>{by_color[c]}</td><td>{color_dim_note[c]}</td></tr>"
        for c in range(C)
    )

    n_pe = kr * kc
    deg_rows = "".join(
        f"<tr><td>{d}</td><td>{cnt}</td><td>{cnt * 100 / n_pe:.1f}%</td></tr>"
        for d, cnt in sorted(deg_hist.items())
    )
    miss_rows = "".join(
        f"<tr><td>{m}</td><td>{cnt}</td></tr>" for m, cnt in sorted(miss_count.items())
    )

    cell = 60 if max(kr, kc) <= 8 else (46 if max(kr, kc) <= 14 else 38)
    margin = 42
    W = margin * 2 + cell * (kc - 1)
    H = margin * 2 + cell * (kr - 1)
    nradius = 14 if max(kr, kc) <= 8 else (11 if max(kr, kc) <= 14 else 9)
    node_font = 10 if n_pe <= 64 else (8 if n_pe <= 200 else 7)
    svg_max = min(1380, W)
    logical_diam = n + 2  # approx upper bound after restriction

    select_options = "".join(
        f'<option value="{c}">C{c} ({by_color[c]} links)</option>' for c in range(C)
    )

    title = f"{kr}×{kc} Mesh · 限制版 (k=2, n={n}) FB · Color 子拓扑"
    out_path = f"docs/tdm_fb_{kr}x{kc}_restricted_color_subtopology.html"

    html = HTML_TEMPLATE.format(
        title=title, parent_rows=pr, parent_cols=pc, n=n,
        full_C=parent_plan.C, C=C,
        full_FB_C=cfg["full_FB_C"], full_FB_links=cfg["full_FB_links"],
        full_FB_degree=cfg["full_FB_degree"],
        intro=cfg["intro"], rule_text=cfg["rule"],
        n_links=n_links, n_pe=n_pe,
        deg_min=min(deg_hist), deg_max=max(deg_hist),
        partial_nodes=partial_nodes,
        logical_diam=logical_diam,
        color_table_rows=color_table_rows, phys_rows=phys_rows,
        deg_rows=deg_rows, miss_rows=miss_rows,
        select_options=select_options,
        links_json=json.dumps(enriched_links, separators=(",", ":")),
        colors_json=json.dumps(colors_meta, separators=(",", ":")),
        partial_nodes_json=json.dumps(partial_renum),
        rows=kr, cols=kc, cell=cell, margin=margin, W=W, H=H,
        nradius=nradius, node_font=node_font, svg_max=svg_max,
        footer_path=out_path,
    )
    Path(out_path).write_text(html, encoding="utf-8")
    return Path(out_path)


def main() -> None:
    for cfg in CONFIGS:
        path = write_html(cfg)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

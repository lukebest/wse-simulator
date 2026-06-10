#!/usr/bin/env python3
"""Generate TDM FB color sub-topology HTML for rectangular meshes.

Targets meshes where (rows, cols) does not fit k^n form: 6x8, 12x16, 14x14.
Uses a 2-flat mixed-radix Flattened Butterfly: each row is fully-connected
intra-row (k=cols), each column fully-connected intra-column (k=rows).
Physical routing is XY-deterministic. Coloring uses the existing greedy
edge-conflict-avoiding scheme in wsesim.network.tdm_coloring.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from wsesim.network.tdm_coloring import assign_colors


PALETTE_BASE = [
    "#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626",
    "#0891b2", "#ca8a04", "#db2777", "#4f46e5", "#059669",
    "#7c3aed", "#c2410c", "#0d9488", "#be123c", "#1d4ed8", "#854d0e",
    "#0369a1", "#65a30d", "#a21caf", "#b91c1c", "#0e7490", "#a16207",
    "#9d174d", "#3730a3", "#047857", "#6d28d9", "#9a3412", "#115e59",
    "#9f1239", "#1e40af", "#713f12", "#075985", "#4d7c0f", "#86198f",
    "#7f1d1d", "#155e75", "#854d0e", "#831843", "#312e81", "#064e3b",
    "#581c87", "#7c2d12", "#134e4a", "#881337", "#1e3a8a", "#422006",
    "#0c4a6e", "#365314", "#701a75", "#7f1d1d", "#164e63", "#78350f",
    "#500724", "#1e1b4b", "#022c22", "#3b0764", "#431407", "#042f2e",
    "#4c0519", "#172554", "#1c1917", "#0a0a0a", "#27272a", "#262626",
]


def build_rect_fb(rows: int, cols: int):
    """Mixed-radix 2-flat FB: row-FB (dim=0) + col-FB (dim=1) with XY paths."""
    n_nodes = rows * cols
    physical: list[tuple[int, int]] = []
    for n in range(n_nodes):
        r, c = divmod(n, cols)
        if c + 1 < cols:
            right = r * cols + c + 1
            physical.append((n, right))
            physical.append((right, n))
        if r + 1 < rows:
            down = (r + 1) * cols + c
            physical.append((n, down))
            physical.append((down, n))

    links: list[tuple[int, int, int]] = []
    paths: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for n in range(n_nodes):
        r, c = divmod(n, cols)
        for c2 in range(cols):
            if c2 == c:
                continue
            v = r * cols + c2
            links.append((n, v, 0))
            step = 1 if c2 > c else -1
            seg: list[tuple[int, int]] = []
            cc = c
            while cc != c2:
                seg.append((r * cols + cc, r * cols + cc + step))
                cc += step
            paths[(n, v)] = seg
        for r2 in range(rows):
            if r2 == r:
                continue
            v = r2 * cols + c
            links.append((n, v, 1))
            step = 1 if r2 > r else -1
            seg = []
            rr = r
            while rr != r2:
                seg.append((rr * cols + c, (rr + step) * cols + c))
                rr += step
            paths[(n, v)] = seg
    return links, paths, physical


def per_distance_breakdown(links, paths, plan, rows, cols):
    """Per-color breakdown by (dim, logical distance) so we can describe structure."""
    table: dict[int, Counter] = {}
    for (u, v), color in plan.color_of_logical.items():
        dim = None
        # Recover dim/dist from coordinates
        ur, uc = divmod(u, cols)
        vr, vc = divmod(v, cols)
        if ur == vr:
            dim, dist = 0, abs(vc - uc)
        else:
            dim, dist = 1, abs(vr - ur)
        table.setdefault(color, Counter())[(dim, dist)] += 1
    return table


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{rows}×{cols} TDM FB · Color 子拓扑</title>
<style>
  :root {{ --bg:#0f1419; --panel:#1a2332; --text:#e8edf4; --muted:#94a3b8; --border:#2d3a4f; --mesh:#334155; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
  header.hero {{ padding:2rem; background:linear-gradient(135deg,#1e293b,#0f172a); border-bottom:1px solid var(--border); }}
  header.hero h1 {{ margin:0 0 .5rem; font-size:1.6rem; }}
  header.hero p {{ margin:.25rem 0; color:var(--muted); max-width:60rem; }}
  main {{ max-width:1400px; margin:0 auto; padding:1.75rem 1.5rem 3rem; }}
  section {{ margin-bottom:2rem; }}
  h2 {{ font-size:1.2rem; margin:0 0 .75rem; }}
  h3 {{ font-size:1rem; margin:1rem 0 .5rem; color:#cbd5e1; }}
  .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:.75rem; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:.85rem 1rem; }}
  .stat .val {{ font-size:1.5rem; font-weight:700; }}
  .stat .lbl {{ color:var(--muted); font-size:.8rem; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:1rem 1.25rem; margin-bottom:1.25rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
  th,td {{ border:1px solid var(--border); padding:.4rem .55rem; text-align:left; }}
  th {{ background:#243044; }}
  .svg-wrap {{ background:#111827; border-radius:8px; padding:.35rem; overflow:auto; }}
  .topo-svg {{ width:100%; max-width:{svg_max}px; height:auto; display:block; margin:0 auto; }}
  .mesh-edge {{ stroke:var(--mesh); stroke-width:1; opacity:.25; pointer-events:none; }}
  .logic-link {{ fill:none; stroke-width:1.2; opacity:.7; cursor:pointer; pointer-events:stroke; stroke-linecap:round; transition:opacity .12s,stroke-width .12s; }}
  .logic-link:hover,.logic-link.hovered {{ opacity:1; stroke-width:2.6; }}
  .logic-link.dimmed {{ opacity:.05; }}
  .node {{ fill:#1e293b; stroke:#64748b; stroke-width:1; pointer-events:none; }}
  .node-label {{ fill:#e2e8f0; font-size:{node_font}px; font-family:ui-monospace,monospace; text-anchor:middle; pointer-events:none; }}
  .filter-bar {{ display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:.5rem; max-height:160px; overflow-y:auto; }}
  .filter-btn {{ border:1px solid var(--border); background:#243044; color:var(--text); padding:.25rem .5rem; border-radius:5px; font-size:.72rem; cursor:pointer; display:flex; align-items:center; gap:.3rem; font-family:ui-monospace,monospace; }}
  .filter-btn.active {{ background:#334155; }}
  .filter-btn.inactive {{ opacity:.35; }}
  .filter-actions {{ display:flex; gap:.5rem; margin-bottom:.6rem; }}
  .filter-actions button {{ background:#1e3a8a; border:1px solid #3b82f6; color:#dbeafe; padding:.3rem .65rem; border-radius:5px; cursor:pointer; font-size:.78rem; }}
  .filter-actions button.alt {{ background:#374151; border-color:#6b7280; color:#e5e7eb; }}
  .dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
  .tooltip {{ position:fixed; z-index:999; pointer-events:none; background:#0f172a; border:1px solid #475569; border-radius:5px; padding:.35rem .55rem; font-family:ui-monospace,monospace; font-size:.78rem; display:none; box-shadow:0 4px 12px rgba(0,0,0,.4); white-space:nowrap; }}
  .tooltip .tag {{ color:#94a3b8; margin-left:.3rem; font-size:.72rem; }}
  .hover-info {{ margin-top:.5rem; font-family:ui-monospace,monospace; font-size:.8rem; color:#cbd5e1; min-height:1.2em; }}
  .color-viewer {{ margin-top:1rem; }}
  .color-viewer select {{ background:#243044; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:.4rem .6rem; font-size:.85rem; }}
  code {{ background:#243044; padding:.1rem .35rem; border-radius:3px; font-size:.85em; }}
  .rule-box {{ font-family:ui-monospace,monospace; background:#111827; border-radius:6px; padding:.65rem 1rem; font-size:.78rem; margin:.5rem 0; line-height:1.55; }}
  details summary {{ cursor:pointer; padding:.4rem 0; font-weight:600; }}
  .breakdown-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:.6rem; font-size:.78rem; }}
  .breakdown-grid .bd-cell {{ background:#0f172a; border:1px solid var(--border); border-radius:6px; padding:.55rem .7rem; }}
  .breakdown-grid .bd-cell strong {{ color:#93c5fd; }}
  footer {{ text-align:center; color:var(--muted); font-size:.75rem; padding:1.5rem; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<header class="hero">
  <h1>{rows}×{cols} 物理 Mesh · 矩形 2-flat Flattened Butterfly · Color 子拓扑分解</h1>
  <p>{n_pe} PE · {n_links} 有向逻辑链路 · TDM <strong>C={C}</strong> · 物理负载下界 = {lb} · 逻辑直径 ≤ 2 跳</p>
  <p>{intro}</p>
</header>
<main>
  <section>
    <h2>核心指标</h2>
    <div class="summary-grid">
      <div class="stat"><div class="val">{C}</div><div class="lbl">TDM Color 数</div></div>
      <div class="stat"><div class="val">{n_links}</div><div class="lbl">有向逻辑链路</div></div>
      <div class="stat"><div class="val">{lb}</div><div class="lbl">物理负载下界</div></div>
      <div class="stat"><div class="val">{n_pe}</div><div class="lbl">PE 数</div></div>
    </div>
  </section>

  <section class="panel">
    <h2>切分方法（结构规则）</h2>
    <div class="rule-box">{rule_text}</div>
    <h3>每条 Color 的逻辑类型分布</h3>
    <p style="color:var(--muted);font-size:.83rem;margin:.25rem 0 .5rem">
      下表展示每条 Color 中所含逻辑链路的 <code>(维度, 行/列距离)</code> 类别。
      <code>R/d</code>=行 FB 距离 d，<code>C/d</code>=列 FB 距离 d。
    </p>
    <div class="breakdown-grid">{breakdown_cells}</div>
  </section>

  <section class="panel">
    <h2>{C} Color 合并预览</h2>
    <div class="filter-actions">
      <button id="select-all">全选</button>
      <button id="deselect-all" class="alt">全不选</button>
      <button id="show-first8" class="alt">仅前 8</button>
    </div>
    <div class="filter-bar" id="filters"></div>
    <div class="svg-wrap" id="merged"></div>
    <div class="hover-info" id="hover-info">将鼠标移到链路上…</div>
  </section>

  <section class="panel color-viewer">
    <h2>单 Color 子拓扑</h2>
    <label>选择 Color：<select id="single-color">{select_options}</select></label>
    <div class="svg-wrap" id="single" style="margin-top:.75rem"></div>
  </section>

  <section class="panel">
    <h2>Color 链路分配（全量表）</h2>
    <details><summary>展开 {C} 条 Color 的链路数</summary>
      <table style="margin-top:.6rem"><thead><tr><th>Color</th><th></th><th>有向链路数</th></tr></thead><tbody>{color_table_rows}</tbody></table>
    </details>
  </section>

  <section class="panel">
    <h2>物理跳数分布</h2>
    <table><thead><tr><th>物理跳数</th><th>链路数</th></tr></thead><tbody>{phys_rows}</tbody></table>
  </section>
</main>
<footer>docs/tdm_fb_{rows}x{cols}_color_subtopology.html · 生成自 scripts/generate_tdm_fb_rect_docs.py</footer>

<script>
const LINKS = {links_json};
const COLORS = {colors_json};
const ROWS={rows}, COLS={cols}, CELL={cell}, MARGIN={margin}, W={W}, H={H}, NRADIUS={nradius};
const INITIAL = {initial_active};

function nodeXY(n) {{ const r=Math.floor(n/COLS), c=n%COLS; return [MARGIN+c*CELL, MARGIN+r*CELL]; }}

function arcPath(src,dst,dim,idx) {{
  const [x0,y0]=nodeXY(src), [x1,y1]=nodeXY(dst);
  const sc=src%COLS, dc=dst%COLS, sr=Math.floor(src/COLS), dr=Math.floor(dst/COLS);
  const dist = dim===0 ? Math.abs(dc-sc) : Math.abs(dr-sr);
  let cx,cy;
  if (dim===0) {{ const s=src<dst?-1:1; cx=(x0+x1)/2; cy=(y0+y1)/2+s*(8+dist*4)*(idx%2?0.7:1); }}
  else {{ const s=src<dst?-1:1; cx=(x0+x1)/2+s*(8+dist*4)*(idx%2?0.7:1); cy=(y0+y1)/2; }}
  return `M ${{x0}} ${{y0}} Q ${{cx}} ${{cy}} ${{x1}} ${{y1}}`;
}}

function markers(ids) {{
  let s='<defs>'; for (const id of ids) s+=`<marker id="a${{id}}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="3.5" markerHeight="3.5" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="${{COLORS[id].hex}}"/></marker>`;
  return s+'</defs>';
}}
function nodes() {{
  let s='';
  for(let n=0;n<ROWS*COLS;n++){{
    const[x,y]=nodeXY(n);
    s+=`<circle cx="${{x}}" cy="${{y}}" r="${{NRADIUS}}" class="node"/><text x="${{x}}" y="${{y+3}}" class="node-label">${{n}}</text>`;
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
let active=new Set(INITIAL);
const btnMap=new Map();
COLORS.forEach(c=>{{
  const b=document.createElement('button');
  b.className='filter-btn '+(active.has(c.id)?'active':'inactive');
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
document.getElementById('show-first8').onclick=()=>{{ active.clear(); COLORS.slice(0,8).forEach(c=>active.add(c.id)); refreshButtons(); render(merged,active,document.getElementById('hover-info')); }};
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
        "rows": 6,
        "cols": 8,
        "intro": (
            "48 = 2⁴·3，无 kⁿ 形式，按矩形 2-flat 混合基 FB 切分：dim0 = 每行 8 节点全连接（K₈），"
            "dim1 = 每列 6 节点全连接（K₆）。XY 维度序路由保证单包跨 ≤2 个逻辑跳。"
        ),
        "rule": (
            "<b>每行 8 节点的行内 FB</b>：行内距离 d 的有向链路共 2·(8−d) 条，"
            "需要 d 个 Color（按源列号 mod d 分桶）。1+2+…+7 = 28 个行内结构 Color，"
            "其中最大同时活跃数 = 4·4 = 16（中间物理边负载下界）。<br>"
            "<b>每列 6 节点的列内 FB</b>：列内距离 d 需要 d 个 Color，列内 Color 总需求 1+…+5 = 15，"
            "最大同时活跃数 = 3·3 = 9。<br>"
            "<b>合并下界</b>：行/列物理边互不冲突，下界 = max(16, 9) = <b>16</b>。"
            "greedy 着色把列 FB 的链路<u>填入行 FB 剩余空闲槽位</u>，最终 <b>C=16=k 行</b>。"
        ),
    },
    {
        "rows": 12,
        "cols": 16,
        "intro": (
            "192 = 2⁶·3，矩形 2-flat 混合基 FB：dim0 = 每行 16 节点全连接（K₁₆），"
            "dim1 = 每列 12 节点全连接（K₁₂）。该尺寸接近 16×16 全 FB 的子集，"
            "C 下界由行 FB 中间边主导。"
        ),
        "rule": (
            "<b>行 FB (K₁₆)</b>：行内距离 d 链路 2·(16−d) 条/行，最大物理边负载 = ⌊16/2⌋·⌈16/2⌉ = 8·8 = <b>64</b>。<br>"
            "<b>列 FB (K₁₂)</b>：列内距离 d 链路 2·(12−d) 条/列，最大物理边负载 = 6·6 = 36。<br>"
            "<b>C = 下界 = 64</b>。Color 结构上可看作：C0..C15 主要承载行 FB 短距离 + 列 FB 全部，"
            "C16..C47 承载行 FB 中距离 (d=3..6)，C48..C63 承载行 FB 长距离 (d=7..15) 与最长列 FB 链路。"
        ),
    },
    {
        "rows": 14,
        "cols": 14,
        "intro": (
            "196 = 14²，正方形 mesh。可直接对应 k=14, n=2 的标准 Flattened Butterfly："
            "dim0 = 每行 14 节点 K₁₄，dim1 = 每列 14 节点 K₁₄。行列对称使 C 下界仅由"
            "单维度行/列中间物理边决定。"
        ),
        "rule": (
            "<b>行/列 FB (K₁₄ 各方向)</b>：距离 d 链路 2·(14−d) 条/线，最大物理边负载 = 7·7 = <b>49</b>。<br>"
            "<b>C = 下界 = 49</b>。结构上行 FB 占 28+ 个 Color、列 FB 占 28+ 个 Color，"
            "因负载分布相同，greedy 把列 FB 全部填进行 FB 留出的 49 个槽位，无需新增。<br>"
            "此例与 (k=14, n=2) 标准 TDMFlatButterfly 等价；行列对称是关键特征。"
        ),
    },
]


def write_html(rows: int, cols: int, intro: str, rule: str) -> Path:
    links, paths, physical = build_rect_fb(rows, cols)
    plan = assign_colors(links, paths, physical)
    C = plan.C
    by_color = Counter(plan.color_of_logical.values())
    by_phys = Counter(len(p) for p in paths.values())
    palette = PALETTE_BASE * ((C // len(PALETTE_BASE)) + 1)

    enriched_links = []
    for u, v, dim in links:
        enriched_links.append({
            "src": u, "dst": v, "dim": dim,
            "color": plan.color_of_logical[(u, v)],
            "phys": len(paths[(u, v)]),
        })

    colors_meta = [{"id": c, "hex": palette[c]} for c in range(C)]
    breakdown = per_distance_breakdown(links, paths, plan, rows, cols)

    def fmt_breakdown(c: int) -> str:
        items = breakdown.get(c, Counter())
        parts = []
        for (dim, dist), n in sorted(items.items()):
            tag = "R" if dim == 0 else "C"
            parts.append(f"{tag}/{dist}×{n}")
        return ", ".join(parts) if parts else "—"

    breakdown_cells = "".join(
        f'<div class="bd-cell"><strong>C{c}</strong> ({by_color[c]}): {fmt_breakdown(c)}</div>'
        for c in range(C)
    )

    color_table_rows = "".join(
        f"<tr><td>C{c}</td><td><span class='dot' style='background:{palette[c]}'></span></td><td>{by_color[c]}</td></tr>"
        for c in range(C)
    )
    phys_rows = "".join(f"<tr><td>{d}</td><td>{by_phys[d]}</td></tr>" for d in sorted(by_phys))

    # Visualization geometry: keep nodes ~36-52 px apart and SVG bounded
    cell = 56 if max(rows, cols) <= 8 else (44 if max(rows, cols) <= 14 else 38)
    margin = 40
    W = margin * 2 + cell * (cols - 1)
    H = margin * 2 + cell * (rows - 1)
    nradius = 13 if max(rows, cols) <= 8 else (11 if max(rows, cols) <= 14 else 9)
    node_font = 9 if rows * cols <= 64 else (8 if rows * cols <= 200 else 7)
    svg_max = min(1380, W)

    select_options = "".join(
        f'<option value="{c}">C{c} ({by_color[c]} links)</option>' for c in range(C)
    )

    # Default-active set: keep initial render light for large meshes.
    initial = list(range(min(C, 8)))

    html = HTML_TEMPLATE.format(
        rows=rows, cols=cols, n_pe=rows * cols, n_links=len(links),
        C=C, lb=plan.color_lower_bound, intro=intro, rule_text=rule,
        breakdown_cells=breakdown_cells,
        color_table_rows=color_table_rows, phys_rows=phys_rows,
        links_json=json.dumps(enriched_links, separators=(",", ":")),
        colors_json=json.dumps(colors_meta, separators=(",", ":")),
        cell=cell, margin=margin, W=W, H=H, nradius=nradius,
        node_font=node_font, svg_max=svg_max,
        select_options=select_options,
        initial_active=json.dumps(initial),
    )
    out = Path(f"docs/tdm_fb_{rows}x{cols}_color_subtopology.html")
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    for cfg in CONFIGS:
        path = write_html(cfg["rows"], cfg["cols"], cfg["intro"], cfg["rule"])
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

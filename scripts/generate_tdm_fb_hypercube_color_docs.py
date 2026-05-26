#!/usr/bin/env python3
"""Generate TDM FB color sub-topology markdown + HTML (ILP min-C coloring).

Covers hypercube FB configs on square meshes:
  4×4: k4_n2, k2_n4
  8×8: k8_n2 (2-flat analogue of k4_n2), k2_n6 (binary multi-flat analogue of k2_n4)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from wsesim.network.color_planners import (
    PLANNER_ILP_MIN_C,
    ColorPlannerConfig,
    build_color_plan,
    plan_stats,
)
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly

PALETTE = [
    "#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626",
    "#0891b2", "#ca8a04", "#db2777", "#4f46e5", "#059669",
    "#7c3aed", "#c2410c", "#0d9488", "#be123c", "#1d4ed8", "#854d0e",
]

CONFIGS = [
    {
        "rows": 4,
        "cols": 4,
        "k": 4,
        "n": 2,
        "slug": "tdm_fb_4x4_color_subtopology",
        "title": "4-ary, 2-flat",
        "logic_hops": 2,
        "neighbors_per_dim": 3,
        "route_pairs": [(0, 9), (0, 15), (0, 8), (7, 8)],
        "compare_note": "4×4 mesh · kⁿ=16 · C=lb=4",
    },
    {
        "rows": 4,
        "cols": 4,
        "k": 2,
        "n": 4,
        "slug": "tdm_fb_k2_n4_color_subtopology",
        "title": "2-ary, 4-flat",
        "logic_hops": 4,
        "neighbors_per_dim": 1,
        "route_pairs": [(0, 5), (0, 6), (0, 15), (3, 12)],
        "compare_note": "4×4 mesh · kⁿ=16 · C=lb=2",
    },
    {
        "rows": 8,
        "cols": 8,
        "k": 8,
        "n": 2,
        "slug": "tdm_fb_8x8_k8_n2_color_subtopology",
        "title": "8-ary, 2-flat",
        "logic_hops": 2,
        "neighbors_per_dim": 7,
        "route_pairs": [(0, 9), (0, 63), (0, 27), (7, 56)],
        "compare_note": "8×8 mesh · 64 PE · 2-flat 高 radix（4×4 k4_n2 的扩展）",
    },
    {
        "rows": 8,
        "cols": 8,
        "k": 2,
        "n": 6,
        "slug": "tdm_fb_8x8_k2_n6_color_subtopology",
        "title": "2-ary, 6-flat",
        "logic_hops": 6,
        "neighbors_per_dim": 1,
        "route_pairs": [(0, 63), (0, 7), (0, 56), (15, 48)],
        "compare_note": "8×8 mesh · 64 PE · binary 多 flat（4×4 k2_n4 的扩展）",
    },
]

ILP_TIME_LIMIT_S = 120.0


def _cell_size(rows: int, cols: int) -> tuple[int, int, int, int, int]:
    cell = 72 if rows <= 4 else 72
    margin = 50
    w = margin * 2 + cell * (cols - 1)
    h = margin * 2 + cell * (rows - 1)
    nr = 14 if rows <= 4 else 14
    return cell, margin, w, h, nr


def build_links(topo: TDMFlatButterfly) -> tuple[list[dict], object, object]:
    config = ColorPlannerConfig(
        planner=PLANNER_ILP_MIN_C,
        topology_hint={"k": topo.k, "n": topo.n},
        time_limit_s=ILP_TIME_LIMIT_S,
    )
    plan, stats = build_color_plan(topo, config)
    links = []
    for u, v, dim in topo.logical_links():
        links.append({
            "src": u,
            "dst": v,
            "dim": dim,
            "color": plan.color_of_logical[(u, v)],
            "phys": len(topo.physical_path(u, v)),
        })
    return links, plan, stats


def route_examples(
    topo: TDMFlatButterfly,
    plan,
    pairs: list[tuple[int, int]],
) -> list[tuple[str, str, str]]:
    rows = []
    for src, dst in pairs:
        hops = topo.dim_order_route(src, dst)
        if not hops:
            continue
        colors = []
        parts = []
        for u, v in hops:
            c = plan.color_of_logical[(u, v)]
            colors.append(c)
            parts.append(f"{u}→{v}(C{c})")
        note = f"单 C{colors[0]}" if len(set(colors)) == 1 else f"跨 {len(set(colors))} Color"
        rows.append((f"{src} → {dst}", " → ".join(parts), note))
    return rows


def node_ref(rows: int, cols: int) -> str:
    lines = []
    for r in range(rows):
        row = [f"{r * cols + c:2d}" for c in range(cols)]
        lines.append(" ".join(row) + f"   ← row {r}")
    return "\n".join(lines)


def write_markdown(cfg: dict, topo: TDMFlatButterfly, plan, stats, links: list[dict]) -> None:
    rows, cols, k, n = cfg["rows"], cfg["cols"], cfg["k"], cfg["n"]
    by_color = Counter(l["color"] for l in links)
    phys_dist = Counter(l["phys"] for l in links)
    routes = route_examples(topo, plan, cfg["route_pairs"])
    mesh = f"{rows}×{cols}"

    color_table = "\n".join(f"| C{c} | {by_color[c]} |" for c in sorted(by_color))
    route_table = "\n".join(f"| `{r}` | {p} | {note} |" for r, p, note in routes)
    phys_rows = "\n".join(f"| {d} | {phys_dist[d]} |" for d in sorted(phys_dist))

    md = f"""# {mesh} Mesh 上 {cfg['title']} Flattened Butterfly 的 Color 子拓扑分解

> **着色策略**：`ilp_min_C`（[`color_planners.py`](../wsesim/network/color_planners.py) · OR-Tools CP-SAT）  
> **目标**：在 {mesh} 物理 2D Mesh 上，通过 **{plan.C} 个 Color** 时分复用实现完整 FB 逻辑拓扑。

---

## 1. 基本定义

### 1.1 节点编号（行优先 {mesh}）

```
{node_ref(rows, cols)}
```

`node = row × {cols} + col`

### 1.2 逻辑拓扑：{cfg['title']} (k={k}, n={n})

- 有向逻辑链路：**{len(links)}**
- 逻辑直径：**≤ {cfg['logic_hops']} 跳**
- 每维邻居数：**{cfg['neighbors_per_dim']}**

---

## 2. Color 分解（ILP min C）

| 指标 | 值 |
|------|-----|
| TDM Color 数 C | **{plan.C}** |
| 物理边负载下界 | {plan.color_lower_bound} |
| 负载均衡 max/min | {stats.balance_ratio:.2f} |
| 每 Color 链路数范围 | {stats.min_links_per_color} – {stats.max_links_per_color} |

### 各 Color 链路分配

| Color | 有向链路数 |
|-------|----------|
{color_table}
| **合计** | **{len(links)}** |

### 物理跳数分布

| 物理跳数 | 链路数 |
|---------|--------|
{phys_rows}

---

## 3. 时分调度

```
周期 C = {plan.C}：时隙 t 激活 Color (t mod {plan.C})
```

---

## 4. 路由示例

| 通信 | 维度序路由 | Color |
|------|-----------|-------|
{route_table}

---

## 5. 说明

{cfg['compare_note']}

---

## 6. 可视化

交互式 Color 时分图：`{cfg['slug']}.html`

---

*生成：`scripts/generate_tdm_fb_hypercube_color_docs.py` · planner={PLANNER_ILP_MIN_C}*
"""
    path = Path(f"docs/{cfg['slug']}.md")
    path.write_text(md, encoding="utf-8")
    print(f"Wrote {path}")


def write_html(cfg: dict, links: list[dict], plan, stats) -> None:
    rows, cols, k, n = cfg["rows"], cfg["cols"], cfg["k"], cfg["n"]
    cell, margin, w, h, nr = _cell_size(rows, cols)
    num_nodes = rows * cols
    c_count = plan.C
    by_color = Counter(l["color"] for l in links)
    colors_meta = [
        {
            "id": c,
            "name": f"Color {c}",
            "hex": PALETTE[c % len(PALETTE)],
            "count": by_color[c],
        }
        for c in range(c_count)
    ]
    links_json = json.dumps(links, separators=(",", ":"))
    colors_json = json.dumps(colors_meta, separators=(",", ":"))
    topo = TDMFlatButterfly(k=k, n=n, rows=rows, cols=cols)
    routes = route_examples(topo, plan, cfg["route_pairs"])
    route_rows = "".join(
        f"<tr><td><code>{r}</code></td><td>{p}</td><td>{note}</td></tr>"
        for r, p, note in routes
    )
    color_table_rows = "".join(
        f"<tr><td>C{c['id']}</td><td><span class='dot' style='background:{c['hex']}'></span></td>"
        f"<td>{c['count']}</td></tr>"
        for c in colors_meta
    )
    select_opts = "".join(
        f'<option value="{c["id"]}">C{c["id"]} ({c["count"]} links)</option>'
        for c in colors_meta
    )
    mesh = f"{rows}×{cols}"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{mesh} TDM FB — {cfg['title']}</title>
<style>
  :root {{ --bg:#0f1419; --panel:#1a2332; --text:#e8edf4; --muted:#94a3b8; --border:#2d3a4f; --mesh:#334155; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
  header.hero {{ padding:2rem; background:linear-gradient(135deg,#1e293b,#0f172a); border-bottom:1px solid var(--border); }}
  header.hero h1 {{ margin:0 0 .5rem; font-size:1.6rem; }}
  header.hero p {{ margin:.25rem 0; color:var(--muted); max-width:54rem; }}
  main {{ max-width:1200px; margin:0 auto; padding:2rem 1.5rem 3rem; }}
  section {{ margin-bottom:2rem; }}
  h2 {{ font-size:1.2rem; margin:0 0 .75rem; }}
  .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:.75rem; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:.85rem 1rem; }}
  .stat .val {{ font-size:1.5rem; font-weight:700; }}
  .stat .lbl {{ color:var(--muted); font-size:.8rem; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:1rem 1.25rem; margin-bottom:1.25rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
  th,td {{ border:1px solid var(--border); padding:.5rem .65rem; text-align:left; }}
  th {{ background:#243044; }}
  .svg-wrap {{ background:#111827; border-radius:8px; padding:.35rem; overflow:auto; }}
  .topo-svg {{ width:100%; max-width:720px; height:auto; display:block; margin:0 auto; }}
  .mesh-edge {{ stroke:var(--mesh); stroke-width:1; opacity:.25; pointer-events:none; }}
  .logic-link {{ fill:none; stroke-width:1.4; opacity:.72; cursor:pointer; pointer-events:stroke; stroke-linecap:round; transition:opacity .12s,stroke-width .12s; }}
  .logic-link:hover,.logic-link.hovered {{ opacity:1; stroke-width:2.8; }}
  .logic-link.dimmed {{ opacity:.06; }}
  .node {{ fill:#1e293b; stroke:#64748b; stroke-width:1.2; pointer-events:none; }}
  .node-label {{ fill:#e2e8f0; font-size:10px; font-family:ui-monospace,monospace; text-anchor:middle; pointer-events:none; }}
  .filter-bar {{ display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:.75rem; max-height:140px; overflow-y:auto; }}
  .filter-btn {{ border:1px solid var(--border); background:#243044; color:var(--text); padding:.3rem .55rem; border-radius:5px; font-size:.75rem; cursor:pointer; display:flex; align-items:center; gap:.3rem; }}
  .filter-btn.active {{ background:#334155; }}
  .filter-btn.inactive {{ opacity:.4; }}
  .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
  .tooltip {{ position:fixed; z-index:999; pointer-events:none; background:#0f172a; border:1px solid #475569; border-radius:5px; padding:.35rem .55rem; font-family:ui-monospace,monospace; font-size:.78rem; display:none; box-shadow:0 4px 12px rgba(0,0,0,.4); white-space:nowrap; }}
  .tooltip .tag {{ color:#94a3b8; margin-left:.3rem; font-size:.72rem; }}
  .hover-info {{ margin-top:.5rem; font-family:ui-monospace,monospace; font-size:.8rem; color:#cbd5e1; min-height:1.2em; }}
  .color-viewer {{ margin-top:1rem; }}
  .color-viewer select {{ background:#243044; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:.4rem .6rem; font-size:.85rem; }}
  code {{ background:#243044; padding:.1rem .35rem; border-radius:3px; font-size:.85em; }}
  footer {{ text-align:center; color:var(--muted); font-size:.75rem; padding:1.5rem; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<header class="hero">
  <h1>{mesh} Mesh · {cfg['title']} Flattened Butterfly (k={k}, n={n})</h1>
  <p>{num_nodes} PE · {len(links)} 有向逻辑链路 · TDM C={c_count} · 着色：<code>{PLANNER_ILP_MIN_C}</code></p>
  <p>负载均衡 max/min = {stats.balance_ratio:.2f}（{stats.min_links_per_color}–{stats.max_links_per_color} links/color）</p>
</header>
<main>
  <section>
    <h2>核心指标</h2>
    <div class="summary-grid">
      <div class="stat"><div class="val">{c_count}</div><div class="lbl">TDM Color 数</div></div>
      <div class="stat"><div class="val">{len(links)}</div><div class="lbl">有向逻辑链路</div></div>
      <div class="stat"><div class="val">{plan.color_lower_bound}</div><div class="lbl">物理负载下界</div></div>
      <div class="stat"><div class="val">≤{cfg['logic_hops']}</div><div class="lbl">逻辑跳数</div></div>
    </div>
  </section>

  <section class="panel">
    <h2>{c_count} Color 合并预览</h2>
    <div class="filter-bar" id="filters"></div>
    <div class="svg-wrap" id="merged"></div>
    <div class="hover-info" id="hover-info">将鼠标移到链路上…</div>
  </section>

  <section class="panel color-viewer">
    <h2>单 Color 子拓扑</h2>
    <label>选择 Color：<select id="single-color">{select_opts}</select></label>
    <div class="svg-wrap" id="single" style="margin-top:.75rem"></div>
  </section>

  <section>
    <h2>Color 链路分配</h2>
    <table><thead><tr><th>Color</th><th></th><th>链路数</th></tr></thead><tbody>{color_table_rows}</tbody></table>
  </section>

  <section>
    <h2>路由示例</h2>
    <table><thead><tr><th>通信</th><th>维度序路由</th><th>Color</th></tr></thead><tbody>{route_rows}</tbody></table>
  </section>
</main>
<footer>docs/{cfg['slug']}.html · ILP min-C · scripts/generate_tdm_fb_hypercube_color_docs.py</footer>

<script>
const LINKS = {links_json};
const COLORS = {colors_json};
const ROWS={rows}, COLS={cols}, CELL={cell}, MARGIN={margin}, W={w}, H={h}, NR={nr};

function nodeXY(n) {{ const r=Math.floor(n/COLS), c=n%COLS; return [MARGIN+c*CELL, MARGIN+r*CELL]; }}

function arcPath(src,dst,dim,idx) {{
  const [x0,y0]=nodeXY(src), [x1,y1]=nodeXY(dst);
  const sc=src%COLS, dc=dst%COLS, sr=Math.floor(src/COLS), dr=Math.floor(dst/COLS);
  const dist = dim===0 ? Math.abs(dc-sc) : Math.abs(dr-sr);
  let cx,cy;
  if (dim===0) {{ const s=src<dst?-1:1; cx=(x0+x1)/2; cy=(y0+y1)/2+s*(10+dist*6)*(idx%2?0.7:1); }}
  else {{ const s=src<dst?-1:1; cx=(x0+x1)/2+s*(10+dist*6)*(idx%2?0.7:1); cy=(y0+y1)/2; }}
  return `M ${{x0}} ${{y0}} Q ${{cx}} ${{cy}} ${{x1}} ${{y1}}`;
}}

function markers(ids) {{
  let s='<defs>'; for (const id of ids) s+=`<marker id="a${{id}}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="${{COLORS[id].hex}}"/></marker>`;
  return s+'</defs>';
}}
function nodes() {{ let s=''; for(let n=0;n<ROWS*COLS;n++){{const[x,y]=nodeXY(n); s+=`<circle cx="${{x}}" cy="${{y}}" r="${{NR}}" class="node"/><text x="${{x}}" y="${{y+4}}" class="node-label">${{n}}</text>`;}} return s; }}
function skeleton() {{ let s=''; for(let n=0;n<ROWS*COLS;n++){{const[x0,y0]=nodeXY(n); const r=Math.floor(n/COLS),c=n%COLS; if(c+1<COLS){{const[x1,y1]=nodeXY(n+1); s+=`<line x1="${{x0}}" y1="${{y0}}" x2="${{x1}}" y2="${{y1}}" class="mesh-edge"/>`;}} if(r+1<ROWS){{const[x1,y1]=nodeXY(n+COLS); s+=`<line x1="${{x0}}" y1="${{y0}}" x2="${{x1}}" y2="${{y1}}" class="mesh-edge"/>`;}} }} return s; }}

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
COLORS.forEach(c=>{{
  const b=document.createElement('button'); b.className='filter-btn active';
  b.innerHTML=`<span class="dot" style="background:${{c.hex}}"></span>C${{c.id}} (${{c.count}})`;
  b.onclick=()=>{{ if(active.has(c.id)){{if(active.size<=1)return; active.delete(c.id); b.classList.replace('active','inactive');}} else{{active.add(c.id); b.classList.replace('inactive','active');}} render(merged,active,document.getElementById('hover-info')); }};
  filters.appendChild(b);
}});
render(merged,active,document.getElementById('hover-info'));

const single=document.getElementById('single');
const sel=document.getElementById('single-color');
function renderSingle(){{ render(single, new Set([+sel.value]), null); }}
sel.onchange=renderSingle; renderSingle();
</script>
</body></html>"""
    path = Path(f"docs/{cfg['slug']}.html")
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    for cfg in CONFIGS:
        topo = TDMFlatButterfly(k=cfg["k"], n=cfg["n"], rows=cfg["rows"], cols=cfg["cols"])
        links, plan, stats = build_links(topo)
        write_markdown(cfg, topo, plan, stats, links)
        write_html(cfg, links, plan, stats)
        print(
            f"  {cfg['rows']}x{cfg['cols']} k={cfg['k']} n={cfg['n']}: "
            f"C={plan.C} balance={stats.balance_ratio:.2f} links={len(links)}"
        )


if __name__ == "__main__":
    main()

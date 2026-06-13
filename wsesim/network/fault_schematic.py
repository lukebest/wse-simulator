"""SVG schematics: golden vs faulty collective routing on a small mesh."""

from __future__ import annotations

import re

from wsesim.network.collective_faults import (
    MeshFault,
    _build_faulty_flows,
    _build_healthy_flows_for_schedule,
    make_scenario_fault,
)
from wsesim.network.collective_patterns import CollectiveFlow, MeshEdge

PATTERNS = (
    "broadcast",
    "reduce",
    "allreduce",
    "allgather",
    "gather",
    "alltoall",
)

PATTERN_TITLES = {
    "broadcast": "Broadcast",
    "reduce": "Reduce",
    "allreduce": "AllReduce",
    "allgather": "AllGather",
    "gather": "Gather",
    "alltoall": "AllToAll",
}

FAULT_TITLES = {
    "golden": "Golden（健康）",
    "pe_point": "PE 坏点",
    "link_point": "链路坏",
    "pe_block": "PE 坏块",
}

FAULT_KEYS: tuple[str, ...] = ("golden", "pe_point", "link_point", "pe_block")


def _cell_center(x: int, y: int, cell: int, pad: int) -> tuple[float, float]:
    return pad + x * cell + cell / 2, pad + y * cell + cell / 2


def _edge_endpoints(edge: MeshEdge, cell: int, pad: int) -> tuple[float, float, float, float]:
    x1, y1 = _cell_center(edge.from_node.x, edge.from_node.y, cell, pad)
    x2, y2 = _cell_center(edge.to_node.x, edge.to_node.y, cell, pad)
    return x1, y1, x2, y2


def _collect_edges(flows: list[CollectiveFlow]) -> set[str]:
    out: set[str] = set()
    for flow in flows:
        for edge in flow.path:
            out.add(edge.id)
    return out


def _sample_flows(
    flows: list[CollectiveFlow],
    pattern: str,
    max_flows: int,
    *,
    cols: int = 4,
) -> list[CollectiveFlow]:
    if len(flows) <= max_flows:
        return flows
    if pattern == "alltoall":
        # Longest detour + corner-to-corner XY baseline
        ranked = sorted(flows, key=lambda f: len(f.path), reverse=True)
        picks = {id(ranked[0])}
        for f in flows:
            if f.from_node.x == 0 and f.from_node.y == 0 and f.to_node.x == cols - 1:
                picks.add(id(f))
            if len(picks) >= max_flows:
                break
        return [f for f in flows if id(f) in picks][:max_flows]
    return flows[:max_flows]


def render_mesh_svg(
    rows: int,
    cols: int,
    flows: list[CollectiveFlow],
    fault: MeshFault,
    *,
    root: int = 0,
    title: str = "",
    highlight_new: set[str] | None = None,
    width: int = 200,
    id_prefix: str = "",
) -> str:
    cell = 36
    pad = 16
    w = pad * 2 + cell * cols
    h = pad * 2 + cell * rows + (18 if title else 0)
    rx, ry = root % cols, root // cols

    mg = f"arr-g{id_prefix}"
    mo = f"arr-o{id_prefix}"
    lines: list[str] = [
        f'<svg viewBox="0 0 {w} {h}" width="{width}" height="{int(width * h / w)}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">',
        "<defs>",
        f'<marker id="{mg}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#4ade80"/></marker>',
        f'<marker id="{mo}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b"/></marker>',
        "</defs>",
    ]
    if title:
        lines.append(
            f'<text x="{w/2}" y="12" text-anchor="middle" fill="#94a3b8" '
            f'font-size="10" font-family="system-ui">{title}</text>'
        )
    y_off = 18 if title else 0

    # grid cells
    for y in range(rows):
        for x in range(cols):
            cx = pad + x * cell
            cy = pad + y * cell + y_off
            bad = (x, y) in fault.bad_nodes
            fill = "#3f1d1d" if bad else "#1e293b"
            stroke = "#f87171" if bad else "#475569"
            lines.append(
                f'<rect x="{cx+2}" y="{cy+2}" width="{cell-4}" height="{cell-4}" '
                f'rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
            if bad:
                lines.append(
                    f'<text x="{cx+cell/2}" y="{cy+cell/2+4}" text-anchor="middle" '
                    f'fill="#f87171" font-size="14" font-weight="bold">×</text>'
                )
            elif x == rx and y == ry:
                lines.append(
                    f'<circle cx="{cx+cell/2}" cy="{cy+cell/2}" r="8" '
                    f'fill="none" stroke="#60a5fa" stroke-width="2"/>'
                )
            lines.append(
                f'<text x="{cx+6}" y="{cy+12}" fill="#64748b" font-size="8">'
                f"{x},{y}</text>"
            )

    # bad links (draw on top of cells)
    drawn_links: set[str] = set()
    for edge_id in fault.bad_edges:
        if edge_id in drawn_links:
            continue
        parts = edge_id.split("->")
        if len(parts) != 2:
            continue
        ax, ay = map(int, parts[0].split(","))
        bx, by = map(int, parts[1].split(","))
        x1, y1 = _cell_center(ax, ay, cell, pad)
        x2, y2 = _cell_center(bx, by, cell, pad)
        y1 += y_off
        y2 += y_off
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#f87171" '
            f'stroke-width="4" stroke-linecap="round" opacity="0.85"/>'
        )
        drawn_links.add(edge_id)

    edge_set = _collect_edges(flows)
    for flow in flows:
        for edge in flow.path:
            x1, y1, x2, y2 = _edge_endpoints(edge, cell, pad)
            y1 += y_off
            y2 += y_off
            is_new = highlight_new and edge.id in highlight_new
            color = "#f59e0b" if is_new else "#4ade80"
            dash = 'stroke-dasharray="4 3"' if is_new else ""
            marker = mg if not is_new else mo
            lines.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                f'stroke-width="2" {dash} marker-end="url(#{marker})" opacity="0.9"/>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


def _fault_for_key(
    rows: int,
    cols: int,
    key: str,
    *,
    demo_region: str = "center",
) -> MeshFault | None:
    if key == "golden":
        return MeshFault()
    return make_scenario_fault(rows, cols, key, demo_region, root=0)  # type: ignore[arg-type]


def flows_for_schematic(
    pattern: str,
    rows: int,
    cols: int,
    fault: MeshFault,
    *,
    root: int = 0,
    golden: bool = False,
) -> list[CollectiveFlow]:
    max_n = 24 if pattern != "alltoall" else 6
    if golden:
        flows = _build_healthy_flows_for_schedule(pattern, rows, cols, root=root)
        return _sample_flows(flows, pattern, max_flows=max_n, cols=cols)
    faulty, meta = _build_faulty_flows(pattern, rows, cols, fault, root=root)
    if not meta.get("reachable", True) and not faulty:
        return []
    return _sample_flows(faulty, pattern, max_flows=max_n, cols=cols)


def render_pattern_row(
    pattern: str,
    rows: int = 4,
    cols: int = 4,
    *,
    demo_region: str = "center",
    root: int = 0,
) -> str:
    """Four-panel row: golden + three fault types."""
    parts: list[str] = [
        f'<div class="pattern-block" id="schem-{pattern}">',
        f'<h3>{PATTERN_TITLES.get(pattern, pattern)}</h3>',
        '<p class="muted schem-note">4×4 示意 · 演示故障位于<strong>中心</strong> '
        f'（root 在 (0,0)）· 绿=原路径 · 橙虚线=绕行/重建</p>',
        '<div class="schem-grid">',
    ]

    golden_flows = flows_for_schematic(
        pattern, rows, cols, MeshFault(), root=root, golden=True
    )
    golden_edges = _collect_edges(golden_flows)

    for key in FAULT_KEYS:
        fault = _fault_for_key(rows, cols, key, demo_region=demo_region)
        assert fault is not None
        if key == "golden":
            flows = golden_flows
            highlight: set[str] | None = None
        else:
            flows = flows_for_schematic(pattern, rows, cols, fault, root=root)
            highlight = _collect_edges(flows) - golden_edges

        caption = FAULT_TITLES[key]
        if key != "golden" and not flows:
            caption += "（不可达）"

        parts.append("<figure class='schem-fig'>")
        parts.append(
            render_mesh_svg(
                rows,
                cols,
                flows,
                fault,
                root=root,
                title=caption,
                highlight_new=highlight if key != "golden" else None,
                id_prefix=f"-{pattern}-{key}",
            )
        )
        parts.append(f"<figcaption>{caption}</figcaption>")
        parts.append("</figure>")

    parts.append("</div></div>")
    return "\n".join(parts)


def render_all_schematics(
    rows: int = 4,
    cols: int = 4,
    *,
    demo_region: str = "center",
) -> str:
    css = """
.schem-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.75rem; margin:1rem 0; }
@media (max-width:1000px) { .schem-grid { grid-template-columns:repeat(2,1fr); } }
@media (max-width:560px) { .schem-grid { grid-template-columns:1fr; } }
.pattern-block { margin-bottom:2.5rem; padding-bottom:1.5rem; border-bottom:1px solid var(--border); }
.pattern-block:last-child { border-bottom:none; }
.schem-fig { margin:0; text-align:center; }
.schem-fig svg { display:block; margin:0 auto; background:#0d1117; border-radius:8px; border:1px solid var(--border); }
.schem-fig figcaption { color:var(--muted); font-size:.8rem; margin-top:.35rem; }
.schem-note { margin:.25rem 0 .75rem; font-size:.85rem; }
"""
    blocks = [
        "<style>",
        css,
        "</style>",
        '<section id="schematics">',
        "<h2>2b. Golden vs 故障处理示意图</h2>",
        '<div class="panel">',
        "<p>每种集合通信在 4×4 mesh 上的<strong>健康（Golden）路由</strong>与三种故障下的"
        "容错处理对比。AllToAll 仅抽样最长/对角线路径以保持可读性。</p>",
    ]
    for pattern in PATTERNS:
        blocks.append(render_pattern_row(pattern, rows, cols, demo_region=demo_region))
    blocks.append("</div></section>")
    return "\n".join(blocks)


def inject_schematics_into_html(html: str, schematics: str) -> str:
    html = re.sub(
        r"<style>\s*\n\.schem-grid[\s\S]*?</section>\s*",
        "",
        html,
        count=1,
    )
    marker = "<section>\n  <h2>3. 退化数据汇总（按 Mesh）</h2>"
    if marker in html:
        return html.replace(marker, schematics + "\n\n" + marker, 1)
    marker2 = "<h2>3. 退化数据汇总"
    if marker2 in html:
        idx = html.index(marker2)
        return html[:idx] + schematics + "\n\n" + html[idx:]
    return html + schematics

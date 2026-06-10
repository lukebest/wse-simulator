"""Visualize and validate TDM color plans on 4x4 mesh (readable version).

Semantics:
- Each color = one conflict-free FB sub-topology slice on the physical mesh.
- C is set by physical edge contention (NOT n).
- For a known (src, dst): pick ONE color shared by ALL logical hops on the
  dimension-order route; the flit keeps that color for its whole lifetime.

How to read the outputs:
- *_overview.png       : physical mesh; edge label = logical FB load (contention).
- *_colors.png           : one subplot per color; curved arrow = logical FB link.
- *_guide.png            : single logical link -> physical XY -> its color.
- *_route_0_5.png        : k2_n4 only; worked example src=0 dst=5.
- *_route_0_6.png        : k2_n4 only; worked example src=0 dst=6.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch

from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly

ROWS = 4
COLS = 4


def _node_xy(node: int, cols: int) -> tuple[float, float]:
    r, c = divmod(node, cols)
    return float(c), float(-r)


def _draw_mesh_grid(ax, rows: int, cols: int, *, label_nodes: bool = True) -> None:
    for node in range(rows * cols):
        x, y = _node_xy(node, cols)
        ax.scatter([x], [y], s=80, c="white", edgecolors="black", linewidths=1.2, zorder=3)
        if label_nodes:
            ax.text(x, y, str(node), ha="center", va="center", fontsize=8, zorder=4)
    # faint physical mesh skeleton
    for node in range(rows * cols):
        r, c = divmod(node, cols)
        x0, y0 = _node_xy(node, cols)
        if c + 1 < cols:
            x1, y1 = _node_xy(node + 1, cols)
            ax.plot([x0, x1], [y0, y1], color="#dddddd", linewidth=1.0, zorder=1)
        if r + 1 < rows:
            x1, y1 = _node_xy(node + cols, cols)
            ax.plot([x0, x1], [y0, y1], color="#dddddd", linewidth=1.0, zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")


def _validate_color_plan(topo: TDMFlatButterfly) -> list[tuple[int, int]]:
    plan = topo.coloring()
    violations: list[tuple[int, int]] = []
    logical_seen: set[tuple[int, int]] = set()
    for logical, color in plan.color_of_logical.items():
        logical_seen.add(logical)
        for edge in topo.physical_path(*logical):
            if plan.link_active[edge][color] != logical:
                violations.append(edge)

    logical_expected = {(u, v) for u, v, _ in topo.logical_links()}
    if logical_seen != logical_expected:
        for logical in logical_expected - logical_seen:
            violations.extend(topo.physical_path(*logical))

    for color in range(plan.C):
        used: set[tuple[int, int]] = set()
        for edge, owners in plan.link_active.items():
            if owners[color] is None:
                continue
            if edge in used:
                violations.append(edge)
            used.add(edge)
    return violations


def _undirected_load(plan, topo: TDMFlatButterfly) -> dict[tuple[int, int], int]:
    """Max load over both directions for cleaner overview."""
    merged: dict[tuple[int, int], int] = {}
    for (src, dst), load in plan.load_per_physical_link.items():
        key = (min(src, dst), max(src, dst))
        merged[key] = max(merged.get(key, 0), load)
    return merged


def _logical_hops(topo: TDMFlatButterfly, src: int, dst: int) -> list[tuple[int, int, int]]:
    """Dimension-order FB route as (u, v, dim) hops."""
    hops: list[tuple[int, int, int]] = []
    for u, v in topo.dim_order_route(src, dst):
        dim = next(
            d
            for d in range(topo.n)
            if topo.to_coords(u)[d] != topo.to_coords(v)[d]
        )
        hops.append((u, v, dim))
    return hops


def _route_color(topo: TDMFlatButterfly, src: int, dst: int) -> int | None:
    return topo.route_color(src, dst)


def _logical_links_in_color(plan, color: int) -> list[tuple[int, int, int]]:
    seen: set[tuple[int, int]] = set()
    links: list[tuple[int, int, int]] = []
    for logical, c in plan.color_of_logical.items():
        if c != color or logical in seen:
            continue
        seen.add(logical)
        dim = plan.logical_dims.get(logical, -1)
        links.append((logical[0], logical[1], dim))
    return sorted(links, key=lambda x: (x[2], x[0], x[1]))


def _draw_logical_arc(
    ax,
    src: int,
    dst: int,
    cols: int,
    *,
    color: str,
    label: str | None = None,
    rad: float = 0.15,
    lw: float = 2.0,
) -> None:
    x0, y0 = _node_xy(src, cols)
    x1, y1 = _node_xy(dst, cols)
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0 + rad * 0.6
        ax.text(mx, my, label, fontsize=7, ha="center", va="center", color=color, zorder=5)


def _plot_overview(topo: TDMFlatButterfly, out_path: Path) -> None:
    plan = topo.coloring()
    loads = _undirected_load(plan, topo)
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_mesh_grid(ax, topo.rows, topo.cols)

    vmax = max(1, max(loads.values()))
    sm = cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=1, vmax=vmax))
    for (a, b), load in loads.items():
        x0, y0 = _node_xy(a, topo.cols)
        x1, y1 = _node_xy(b, topo.cols)
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=sm.to_rgba(load),
            linewidth=2.0 + 2.5 * load / vmax,
            zorder=2,
        )
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        ax.text(mx, my, str(load), fontsize=9, ha="center", va="center", color="black", zorder=5)

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("load(e) = # logical FB paths through this physical link")
    ax.set_title(
        f"Physical 4x4 mesh load map | k={topo.k}, n={topo.n}\n"
        f"C={plan.C} colors needed (lower bound={plan.color_lower_bound})"
    )
    fig.text(
        0.5,
        0.02,
        "Black numbers = node id (row-major). Edge number = contention, NOT color id.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_colors(topo: TDMFlatButterfly, out_path: Path) -> None:
    plan = topo.coloring()
    cols = min(4, plan.C)
    rows = ceil(plan.C / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    axes_list = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]
    palette = plt.get_cmap("tab10", max(plan.C, 1))

    for color in range(plan.C):
        ax = axes_list[color]
        _draw_mesh_grid(ax, topo.rows, topo.cols)
        logical_links = _logical_links_in_color(plan, color)
        draw_color = palette(color)
        # alternate arc curvature so opposite directions remain visible
        for idx, (src, dst, dim) in enumerate(logical_links):
            rad = 0.12 if idx % 2 == 0 else -0.12
            if src // topo.cols == dst // topo.cols:
                rad = 0.18 if src < dst else -0.18
            _draw_logical_arc(
                ax,
                src,
                dst,
                topo.cols,
                color=draw_color,
                label=f"{src}->{dst} d{dim}",
                rad=rad,
            )
        ax.set_title(f"color {color}: {len(logical_links)} logical FB links", fontsize=10)

    for idx in range(plan.C, len(axes_list)):
        axes_list[idx].axis("off")

    fig.suptitle(
        f"Each subplot = one TDM color = conflict-free FB sub-topology on physical mesh.\n"
        f"k={topo.k}, n={topo.n}, C={plan.C} (physical lower bound={plan.color_lower_bound}, NOT n).",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_guide(topo: TDMFlatButterfly, out_path: Path) -> None:
    """Single annotated example: one logical link, its XY path, and its color."""
    plan = topo.coloring()
    # pick a visible horizontal link 0 -> 2 (dim 0)
    src, dst = 0, 2
    logical = (src, dst)
    color = plan.color_of_logical[logical]
    path = topo.physical_path(src, dst)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) logical link
    ax = axes[0]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    _draw_logical_arc(ax, src, dst, topo.cols, color="tab:blue", label=f"logical {src}->{dst}", rad=0.2)
    ax.set_title("(a) One logical FB link\n(same row/col group, dim 0)")

    # (b) physical XY path
    ax = axes[1]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    for hop_src, hop_dst in path:
        x0, y0 = _node_xy(hop_src, topo.cols)
        x1, y1 = _node_xy(hop_dst, topo.cols)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="tab:orange", lw=2.5),
            zorder=2,
        )
    ax.set_title(f"(b) Physical XY path\n{path} (2 hops on mesh)")

    # (c) which color owns it
    ax = axes[2]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    for hop_src, hop_dst in path:
        x0, y0 = _node_xy(hop_src, topo.cols)
        x1, y1 = _node_xy(hop_dst, topo.cols)
        ax.plot([x0, x1], [y0, y1], color="tab:green", linewidth=3, zorder=2)
    ax.set_title(f"(c) Active only in color {color}\n(other colors: these hops idle)")

    fig.suptitle(
        f"Reading guide for k={topo.k}, n={topo.n}: logical link -> physical hops -> TDM color",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_route_guide(topo: TDMFlatButterfly, src: int, dst: int, out_path: Path) -> None:
    """Worked example: pick one color for a multi-hop src->dst route."""
    plan = topo.coloring()
    hops = _logical_hops(topo, src, dst)
    route_color_val = _route_color(topo, src, dst)
    if route_color_val is None:
        print(f"[WARN] default plan has no single color for {src}->{dst}; skip {out_path.name}")
        return

    palette = plt.get_cmap("tab10", max(plan.C, 1))
    slot_color = palette(route_color_val)
    hop_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    ax = axes[0]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    for i, (u, v, dim) in enumerate(hops):
        c = hop_colors[i % len(hop_colors)]
        _draw_logical_arc(ax, u, v, topo.cols, color=c, label=f"d{dim}: {u}->{v}", rad=0.22 - 0.08 * i)
    sx, sy = _node_xy(src, topo.cols)
    dx, dy = _node_xy(dst, topo.cols)
    ax.scatter([sx], [sy], s=220, facecolors="none", edgecolors="tab:blue", linewidths=2, zorder=6)
    ax.scatter([dx], [dy], s=220, facecolors="none", edgecolors="tab:red", linewidths=2, zorder=6)
    hop_labels = " -> ".join(str(u) for u, _, _ in hops) + f" -> {dst}"
    ax.set_title(
        f"(a) Logical route {src} -> {dst}\n"
        f"{len(hops)} FB hops (<= n={topo.n}): {hop_labels}"
    )

    ax = axes[1]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    for idx, (u, v, dim) in enumerate(_logical_links_in_color(plan, route_color_val)):
        rad = 0.10 if idx % 2 == 0 else -0.10
        _draw_logical_arc(ax, u, v, topo.cols, color=slot_color, rad=rad, lw=1.2)
    for u, v, dim in hops:
        _draw_logical_arc(
            ax, u, v, topo.cols, color="black", label=f"{u}->{v} d{dim}", rad=0.25, lw=2.8,
        )
    ax.set_title(
        f"(b) Pick color = {route_color_val} for whole lifetime\n"
        f"all {len(hops)} hops belong to color {route_color_val} sub-topology"
    )

    ax = axes[2]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    phys_edges: list[tuple[int, int]] = []
    for u, v, _ in hops:
        for hop_src, hop_dst in topo.physical_path(u, v):
            x0, y0 = _node_xy(hop_src, topo.cols)
            x1, y1 = _node_xy(hop_dst, topo.cols)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2.8),
                zorder=2,
            )
            phys_edges.append((hop_src, hop_dst))
    ax.set_title(
        f"(c) Physical XY hops (sequential, no shared edge)\n"
        f"{phys_edges}  ->  non-blocking within color {route_color_val}"
    )

    fig.suptitle(
        f"k={topo.k}, n={topo.n}: {src} -> {dst} uses color {route_color_val} "
        f"(constant, <= {len(hops)} logical hops)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_route_0_6_guide(topo: TDMFlatButterfly, out_path: Path) -> None:
    """0->6: default greedy plan fails; show min-C path-aware fix."""
    from wsesim.network.tdm_coloring import assign_colors_for_routes

    src, dst = 0, 6
    plan = topo.coloring()
    hops = _logical_hops(topo, src, dst)
    path_links = topo.dim_order_route(src, dst)
    greedy_colors = [plan.color_of_logical[(u, v)] for u, v, _ in hops]
    min_c, fixed_color = topo.min_route_color(src, dst)

    links = topo.logical_links()
    paths = {(u, v): topo.physical_path(u, v) for u, v, _ in links}
    fixed_plan = assign_colors_for_routes(
        links, paths, topo.physical_links(), force_monochrome_paths=[path_links]
    )

    hop_colors = ["#1f77b4", "#ff7f0e"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    ax = axes[0]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    for i, (u, v, dim) in enumerate(hops):
        _draw_logical_arc(ax, u, v, topo.cols, color=hop_colors[i], label=f"d{dim}: {u}->{v}", rad=0.22 - 0.08 * i)
    ax.set_title(f"(a) Logical route 0->6\n0->2->6 ({len(hops)} hops)")

    ax = axes[1]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    for i, (u, v, dim) in enumerate(hops):
        c = hop_colors[i]
        _draw_logical_arc(ax, u, v, topo.cols, color=c, label=f"color {greedy_colors[i]}", rad=0.22 - 0.08 * i)
    ax.set_title(
        f"(b) Default greedy plan: NO valid pick\n"
        f"hop colors = {greedy_colors} (must be equal)"
    )

    ax = axes[2]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    slot = plt.get_cmap("tab10")(fixed_color if fixed_color is not None else 0)
    for idx, (u, v, dim) in enumerate(_logical_links_in_color(fixed_plan, fixed_color or 0)):
        _draw_logical_arc(ax, u, v, topo.cols, color=slot, rad=0.10 if idx % 2 == 0 else -0.10, lw=1.0)
    for u, v, dim in hops:
        _draw_logical_arc(ax, u, v, topo.cols, color="black", label=f"{u}->{v}", rad=0.25, lw=2.8)
    ax.set_title(
        f"(c) Path-aware fix: min C={min_c}, pick color {fixed_color}\n"
        f"both hops share color {fixed_color}"
    )

    ax = axes[3]
    _draw_mesh_grid(ax, topo.rows, topo.cols)
    phys: list[tuple[int, int]] = []
    for u, v, _ in hops:
        for a, b in topo.physical_path(u, v):
            x0, y0 = _node_xy(a, topo.cols)
            x1, y1 = _node_xy(b, topo.cols)
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2.8), zorder=2)
            phys.append((a, b))
    ax.set_title(f"(d) Physical hops {phys}\nedge-disjoint -> non-blocking")

    fig.suptitle(
        f"k=2,n=4: 0->6 needs path-aware coloring; min C={min_c}, choose color {fixed_color}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary(topo: TDMFlatButterfly, out_path: Path) -> None:
    plan = topo.coloring()
    lines = [
        f"k={topo.k}, n={topo.n}, C={plan.C}, lower_bound={plan.color_lower_bound}",
        "Each color = conflict-free FB sub-topology (C from physical mesh, NOT n).",
        "Route rule: pick ONE color shared by all logical hops on the dim-order path.",
        "",
        "Node layout (row-major 4x4):",
        " 0  1  2  3",
        " 4  5  6  7",
        " 8  9 10 11",
        "12 13 14 15",
        "",
        "Per-color logical link count:",
    ]
    for color in range(plan.C):
        links = _logical_links_in_color(plan, color)
        lines.append(f"  color {color}: {len(links)} links")
        for src, dst, dim in links[:6]:
            hops = topo.physical_path(src, dst)
            lines.append(f"    {src}->{dst} dim{dim}, XY hops={len(hops)}")
        if len(links) > 6:
            lines.append(f"    ... {len(links) - 6} more")
    if topo.k == 2 and topo.n == 4:
        for src, dst in ((0, 5), (0, 6)):
            hops = _logical_hops(topo, src, dst)
            c = _route_color(topo, src, dst)
            lines.extend(["", f"Example {src} -> {dst}:", f"  logical hops = {hops}"])
            if c is not None:
                lines.append(f"  default plan: pick color = {c}")
            else:
                min_c, fixed = topo.min_route_color(src, dst)
                lines.append("  default plan: NO single color (hops disagree)")
                lines.append(f"  path-aware min C={min_c}, pick color = {fixed}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_case(k: int, n: int, out_dir: Path) -> None:
    topo = TDMFlatButterfly(k=k, n=n, rows=ROWS, cols=COLS)
    violations = _validate_color_plan(topo)
    if violations:
        print(f"[WARN] k={k}, n={n}, violations={len(violations)}")
    _plot_overview(topo, out_dir / f"k{k}_n{n}_overview.png")
    _plot_colors(topo, out_dir / f"k{k}_n{n}_colors.png")
    _plot_guide(topo, out_dir / f"k{k}_n{n}_guide.png")
    if k == 2 and n == 4:
        _plot_route_guide(topo, 0, 5, out_dir / f"k{k}_n{n}_route_0_5.png")
        _plot_route_0_6_guide(topo, out_dir / f"k{k}_n{n}_route_0_6.png")
    _write_summary(topo, out_dir / f"k{k}_n{n}_summary.txt")


def main() -> None:
    out_dir = Path("outputs/tdm_flatbf_4x4/coloring")
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, n in ((4, 2), (2, 4)):
        _render_case(k, n, out_dir)
    print(f"wrote readable coloring images to {out_dir}")


if __name__ == "__main__":
    main()

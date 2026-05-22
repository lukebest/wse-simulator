"""Visualize end-to-end TDM colors on 4x4 mesh — logical view only.

Draws ONLY logical end-to-end connections (one arc per src->dst pair).
No physical mesh hops, no XY routing detail.

Outputs (outputs/tdm_flatbf_4x4/coloring_e2e/):
- e2e_colors.png  : one subplot per color; each arc = one logical src->dst pair.
- e2e_guide.png   : worked example for 0 -> 6 (logical hops + assigned color).
- e2e_summary.txt : text listing.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from wsesim.network.tdm_e2e_coloring import assign_e2e_colors
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly

ROWS = 4
COLS = 4
K, N = 2, 4  # k=2, n=4 for 4x4 FB coordinates shown in guide


def _node_xy(node: int) -> tuple[float, float]:
    r, c = divmod(node, COLS)
    return float(c), float(-r)


def _draw_nodes(ax) -> None:
    """Nodes only — no physical mesh edges."""
    for node in range(ROWS * COLS):
        x, y = _node_xy(node)
        ax.scatter([x], [y], s=90, c="white", edgecolors="black", linewidths=1.4, zorder=3)
        ax.text(x, y, str(node), ha="center", va="center", fontsize=9, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.6, COLS - 0.4)
    ax.set_ylim(-ROWS + 0.4, 0.6)


def _arc_rad(src: int, dst: int, idx: int) -> float:
    """Pick curvature so opposite pairs stay visible."""
    if src == dst:
        return 0.0
    sr, sc = divmod(src, COLS)
    dr, dc = divmod(dst, COLS)
    base = 0.22 if idx % 2 == 0 else -0.22
    if sr == dr:
        return 0.28 if sc < dc else -0.28
    if sc == dc:
        return 0.28 if sr < dr else -0.28
    return base


def _draw_logical_arc(
    ax,
    src: int,
    dst: int,
    color,
    *,
    label: str | None = None,
    rad: float = 0.2,
    lw: float = 1.8,
) -> None:
    x0, y0 = _node_xy(src)
    x1, y1 = _node_xy(dst)
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
    )
    if label:
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0 + rad * 0.55
        ax.text(mx, my, label, fontsize=6, ha="center", va="center", color=color, zorder=5)


def _logical_hops(topo: TDMFlatButterfly, src: int, dst: int) -> list[tuple[int, int, int]]:
    """Dimension-order logical hops: (u, v, dim) with at most n hops."""
    if src == dst:
        return []
    hops: list[tuple[int, int, int]] = []
    cur = src
    src_coords = list(topo.to_coords(cur))
    dst_coords = topo.to_coords(dst)
    for dim in range(topo.n):
        if src_coords[dim] == dst_coords[dim]:
            continue
        nxt_coords = list(src_coords)
        nxt_coords[dim] = dst_coords[dim]
        nxt = topo.to_node(tuple(nxt_coords))
        hops.append((cur, nxt, dim))
        cur = nxt
        src_coords = nxt_coords
    return hops


def _plot_colors(plan, out_path: Path) -> None:
    cols = 4
    rows = ceil(plan.C / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows))
    axes_list = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]
    palette = plt.get_cmap("tab20", max(plan.C, 1))

    for color in range(plan.C):
        ax = axes_list[color]
        _draw_nodes(ax)
        draw_color = palette(color)
        pairs = sorted(plan.pairs_by_color[color])
        for idx, (src, dst) in enumerate(pairs):
            _draw_logical_arc(
                ax,
                src,
                dst,
                draw_color,
                label=f"{src}->{dst}",
                rad=_arc_rad(src, dst, idx),
            )
        ax.set_title(f"color {color}: {len(pairs)} logical pairs", fontsize=10)

    for idx in range(plan.C, len(axes_list)):
        axes_list[idx].axis("off")

    fig.suptitle(
        "End-to-end TDM coloring (logical view only)\n"
        "Each curved arrow = one src->dst pair sharing this color for its whole lifetime.\n"
        "No physical mesh — only logical connections.",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_guide(plan, topo: TDMFlatButterfly, out_path: Path) -> None:
    src, dst = 0, 6
    color = plan.color_of_pair[(src, dst)]
    hops = _logical_hops(topo, src, dst)
    mates = sorted(plan.pairs_by_color[color])
    hop_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # (a) logical route broken into FB dimension hops
    ax = axes[0]
    _draw_nodes(ax)
    for i, (u, v, dim) in enumerate(hops):
        c = hop_colors[i % len(hop_colors)]
        _draw_logical_arc(ax, u, v, c, label=f"d{dim}: {u}->{v}", rad=0.25 - 0.1 * i, lw=2.5)
    sx, sy = _node_xy(src)
    dx, dy = _node_xy(dst)
    ax.scatter([sx], [sy], s=200, facecolors="none", edgecolors="tab:blue", linewidths=2, zorder=6)
    ax.scatter([dx], [dy], s=200, facecolors="none", edgecolors="tab:red", linewidths=2, zorder=6)
    ax.set_title(
        f"(a) Logical route {src} -> {dst}\n"
        f"{len(hops)} FB hops (max n={topo.n}); coords {topo.to_coords(src)} -> {topo.to_coords(dst)}"
    )

    # (b) same color: all pairs as single logical arcs
    ax = axes[1]
    _draw_nodes(ax)
    palette = plt.get_cmap("tab20", max(plan.C, 1))
    c = palette(color)
    for idx, (s, d) in enumerate(mates):
        lw = 2.8 if (s, d) == (src, dst) else 1.4
        _draw_logical_arc(ax, s, d, c, label=f"{s}->{d}" if (s, d) == (src, dst) else None, rad=_arc_rad(s, d, idx), lw=lw)
    ax.set_title(f"(b) color {color}: {len(mates)} logical pairs\n(thick = example {src}->{dst})")

    fig.suptitle(
        f"k={topo.k}, n={topo.n} — end-to-end color of {src}->{dst} is {color}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary(plan, topo: TDMFlatButterfly, out_path: Path) -> None:
    lines = [
        f"End-to-end logical coloring on 4x4 mesh (k={topo.k}, n={topo.n})",
        f"C = {plan.C} colors (physical lower bound = {plan.color_lower_bound})",
        "Visualization: one arc per src->dst (logical connection only, no physical hops).",
        "",
        "Node layout:",
        " 0  1  2  3",
        " 4  5  6  7",
        " 8  9 10 11",
        "12 13 14 15",
        "",
        "Example 0 -> 6:",
        f"  color = {plan.color_of_pair[(0, 6)]}",
        f"  logical hops = {_logical_hops(topo, 0, 6)}",
        "",
        "Pairs per color:",
    ]
    for color in range(plan.C):
        pairs = sorted(plan.pairs_by_color[color])
        lines.append(f"  color {color}: {len(pairs)} pairs")
        sample = ", ".join(f"{s}->{d}" for s, d in pairs[:10])
        lines.append(f"    {sample}" + (" ..." if len(pairs) > 10 else ""))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = Path("outputs/tdm_flatbf_4x4/coloring_e2e")
    out_dir.mkdir(parents=True, exist_ok=True)
    topo = TDMFlatButterfly(k=K, n=N, rows=ROWS, cols=COLS)
    plan = assign_e2e_colors(ROWS, COLS)
    _plot_colors(plan, out_dir / "e2e_colors.png")
    _plot_guide(plan, topo, out_dir / "e2e_guide.png")
    _write_summary(plan, topo, out_dir / "e2e_summary.txt")
    # remove stale physical-view overview if present
    stale = out_dir / "e2e_overview.png"
    if stale.exists():
        stale.unlink()
    print(f"wrote logical-only coloring to {out_dir}")


if __name__ == "__main__":
    main()

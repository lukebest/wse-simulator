"""Visualize end-to-end TDM color plan on a 4x4 mesh.

Semantics (per user):
- One color per (src, dst); flit uses that color for its full lifetime.
- Within a color, all active (src, dst) end-to-end XY paths are physically
  edge-disjoint on the mesh.
- Max logical hops = n (whole XY route on the mesh).

Generated files (outputs/tdm_flatbf_4x4/coloring_e2e/):
- e2e_overview.png : physical mesh + edge load(e) (# pairs whose XY path uses e).
- e2e_colors.png   : one subplot per color; arrows = end-to-end paths in that color.
- e2e_guide.png    : worked example for 0 -> 6.
- e2e_summary.txt  : color count, pairs per color, sample listings.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch

from wsesim.network.tdm_e2e_coloring import assign_e2e_colors, physical_xy_path

ROWS = 4
COLS = 4


def _node_xy(node: int) -> tuple[float, float]:
    r, c = divmod(node, COLS)
    return float(c), float(-r)


def _draw_mesh(ax) -> None:
    for node in range(ROWS * COLS):
        x, y = _node_xy(node)
        ax.scatter([x], [y], s=80, c="white", edgecolors="black", linewidths=1.2, zorder=3)
        ax.text(x, y, str(node), ha="center", va="center", fontsize=8, zorder=4)
    for node in range(ROWS * COLS):
        r, c = divmod(node, COLS)
        x0, y0 = _node_xy(node)
        if c + 1 < COLS:
            x1, y1 = _node_xy(node + 1)
            ax.plot([x0, x1], [y0, y1], color="#dddddd", linewidth=1.0, zorder=1)
        if r + 1 < ROWS:
            x1, y1 = _node_xy(node + COLS)
            ax.plot([x0, x1], [y0, y1], color="#dddddd", linewidth=1.0, zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_path(ax, src: int, dst: int, color, *, label: str | None = None) -> None:
    path = physical_xy_path(src, dst, ROWS, COLS)
    for hop_src, hop_dst in path:
        x0, y0 = _node_xy(hop_src)
        x1, y1 = _node_xy(hop_dst)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, shrinkA=6, shrinkB=6),
            zorder=2,
        )
    # arc label near the midpoint of source->dest straight line
    sx, sy = _node_xy(src)
    dx, dy = _node_xy(dst)
    mx, my = (sx + dx) / 2, (sy + dy) / 2
    ax.text(
        mx,
        my + 0.18,
        f"{src}->{dst}" if label is None else label,
        fontsize=6,
        color=color,
        ha="center",
        va="center",
        zorder=5,
    )


def _plot_overview(plan, out_path: Path) -> None:
    merged: dict[tuple[int, int], int] = {}
    for (a, b), load in plan.load_per_physical_link.items():
        key = (min(a, b), max(a, b))
        merged[key] = max(merged.get(key, 0), load)

    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_mesh(ax)
    vmax = max(1, max(merged.values()))
    sm = cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=1, vmax=vmax))
    for (a, b), load in merged.items():
        x0, y0 = _node_xy(a)
        x1, y1 = _node_xy(b)
        ax.plot([x0, x1], [y0, y1], color=sm.to_rgba(load), linewidth=2.0 + 2.5 * load / vmax, zorder=2)
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        ax.text(mx, my, str(load), fontsize=9, ha="center", va="center", zorder=5)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("load(e) = # src->dst pairs whose XY path uses e")
    ax.set_title(
        f"End-to-end coloring: physical load on 4x4 mesh\n"
        f"C={plan.C} colors (lower bound={plan.color_lower_bound}); 240 directed pairs total"
    )
    fig.text(
        0.5,
        0.02,
        "Number on each edge = contention. Same number bigger -> more contention -> needs more TDM colors.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_colors(plan, out_path: Path) -> None:
    cols = 4
    rows = ceil(plan.C / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows))
    axes_list = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]
    palette = plt.get_cmap("tab20", max(plan.C, 1))

    for color in range(plan.C):
        ax = axes_list[color]
        _draw_mesh(ax)
        draw_color = palette(color)
        pairs = sorted(plan.pairs_by_color[color])
        for src, dst in pairs:
            _draw_path(ax, src, dst, draw_color)
        ax.set_title(f"color {color}: {len(pairs)} src->dst pairs", fontsize=10)

    for idx in range(plan.C, len(axes_list)):
        axes_list[idx].axis("off")

    fig.suptitle(
        "End-to-end TDM coloring on 4x4 mesh.\n"
        "Each subplot = one color = one set of edge-disjoint full XY paths.\n"
        "A flit uses ONE color for its entire src->dst lifetime; max logical hops = n.",
        fontsize=11,
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_guide(plan, out_path: Path) -> None:
    src, dst = 0, 6
    color = plan.color_of_pair[(src, dst)]
    path = plan.path_of_pair[(src, dst)]
    color_mates = sorted(plan.pairs_by_color[color])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # (a) src and dst marked
    ax = axes[0]
    _draw_mesh(ax)
    for node, label, c in ((src, "src", "tab:blue"), (dst, "dst", "tab:red")):
        x, y = _node_xy(node)
        ax.scatter([x], [y], s=260, facecolors="none", edgecolors=c, linewidths=2.2, zorder=5)
        ax.text(x, y - 0.35, label, ha="center", color=c, fontsize=10)
    ax.set_title(f"(a) Pair to send: {src} -> {dst}")

    # (b) full XY path drawn as arrows
    ax = axes[1]
    _draw_mesh(ax)
    _draw_path(ax, src, dst, "tab:green", label=f"{src}->{dst} (3 mesh hops)")
    ax.set_title(f"(b) XY path: {path}\nUsed for the whole lifetime")

    # (c) other pairs sharing this color
    ax = axes[2]
    _draw_mesh(ax)
    palette = plt.get_cmap("tab20", max(plan.C, 1))
    chosen_color = palette(color)
    for s, d in color_mates:
        _draw_path(ax, s, d, chosen_color)
    ax.set_title(
        f"(c) color {color}: {len(color_mates)} pairs share this color\n"
        f"all edge-disjoint, can fire together"
    )

    fig.suptitle(
        f"k=2, n=4 (4x4 mesh) - end-to-end coloring guide for {src} -> {dst}: color = {color}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_summary(plan, out_path: Path) -> None:
    lines = [
        f"End-to-end coloring on 4x4 mesh (k,n agnostic; uses XY routing)",
        f"C = {plan.C} colors (lower bound = {plan.color_lower_bound})",
        f"total directed pairs colored = {sum(len(v) for v in plan.pairs_by_color.values())}",
        "",
        "Node layout (row-major):",
        " 0  1  2  3",
        " 4  5  6  7",
        " 8  9 10 11",
        "12 13 14 15",
        "",
        "Pairs per color:",
    ]
    for color in range(plan.C):
        pairs = sorted(plan.pairs_by_color[color])
        lines.append(f"  color {color}: {len(pairs)} pairs")
        for src, dst in pairs[:8]:
            lines.append(f"    {src}->{dst}  XY={plan.path_of_pair[(src, dst)]}")
        if len(pairs) > 8:
            lines.append(f"    ... {len(pairs) - 8} more")
    lines.append("")
    lines.append("Example query: color of (0 -> 6) = " + str(plan.color_of_pair[(0, 6)]))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = Path("outputs/tdm_flatbf_4x4/coloring_e2e")
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = assign_e2e_colors(ROWS, COLS)
    _plot_overview(plan, out_dir / "e2e_overview.png")
    _plot_colors(plan, out_dir / "e2e_colors.png")
    _plot_guide(plan, out_dir / "e2e_guide.png")
    _write_summary(plan, out_dir / "e2e_summary.txt")
    print(f"wrote end-to-end coloring outputs to {out_dir}")


if __name__ == "__main__":
    main()

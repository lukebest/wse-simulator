"""Visualize and validate TDM color plans on 8x8 mesh."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm

from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly


def _node_xy(node: int, cols: int) -> tuple[float, float]:
    r, c = divmod(node, cols)
    return float(c), float(-r)


def _draw_mesh_nodes(ax, rows: int, cols: int) -> None:
    xs = []
    ys = []
    for node in range(rows * cols):
        x, y = _node_xy(node, cols)
        xs.append(x)
        ys.append(y)
    ax.scatter(xs, ys, s=10, c="black", alpha=0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def _validate_color_plan(topo: TDMFlatButterfly) -> list[tuple[int, int]]:
    plan = topo.coloring()
    violations: list[tuple[int, int]] = []
    logical_seen: set[tuple[int, int]] = set()
    for logical, color in plan.color_of_logical.items():
        logical_seen.add(logical)
        for edge in topo.physical_path(*logical):
            active = plan.link_active[edge][color]
            if active != logical:
                violations.append(edge)

    logical_expected = {(u, v) for u, v, _ in topo.logical_links()}
    if logical_seen != logical_expected:
        missing = list(logical_expected - logical_seen)
        for logical in missing:
            for edge in topo.physical_path(*logical):
                violations.append(edge)

    for color in range(plan.C):
        used: set[tuple[int, int]] = set()
        for edge, owners in plan.link_active.items():
            owner = owners[color]
            if owner is None:
                continue
            if edge in used:
                violations.append(edge)
            used.add(edge)
    return violations


def _plot_overview(topo: TDMFlatButterfly, out_path: Path, violations: list[tuple[int, int]]) -> None:
    plan = topo.coloring()
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_mesh_nodes(ax, topo.rows, topo.cols)
    vmax = max(1, max(plan.load_per_physical_link.values()))
    for edge, load in plan.load_per_physical_link.items():
        src, dst = edge
        x0, y0 = _node_xy(src, topo.cols)
        x1, y1 = _node_xy(dst, topo.cols)
        color = cm.viridis(load / vmax)
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.0 + 2.0 * load / vmax, alpha=0.9)
    for edge in violations:
        src, dst = edge
        x0, y0 = _node_xy(src, topo.cols)
        x1, y1 = _node_xy(dst, topo.cols)
        ax.plot([x0, x1], [y0, y1], color="red", linewidth=3, alpha=0.8)
    ax.set_title(
        f"k={topo.k}, n={topo.n} | C={plan.C}, lower_bound={plan.color_lower_bound}, violations={len(violations)}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_colors(topo: TDMFlatButterfly, out_path: Path, violations: list[tuple[int, int]]) -> None:
    plan = topo.coloring()
    cols = 4
    rows = ceil(plan.C / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes_list = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]
    palette = plt.get_cmap("tab20", max(plan.C, 1))

    for color in range(plan.C):
        ax = axes_list[color]
        _draw_mesh_nodes(ax, topo.rows, topo.cols)
        draw_color = palette(color)
        for edge, owners in plan.link_active.items():
            owner = owners[color]
            if owner is None:
                continue
            src, dst = edge
            x0, y0 = _node_xy(src, topo.cols)
            x1, y1 = _node_xy(dst, topo.cols)
            ax.plot([x0, x1], [y0, y1], color=draw_color, linewidth=2.5, alpha=0.9)
            dim = plan.logical_dims.get(owner, -1)
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ax.text(mx, my, f"{owner[0]}->{owner[1]}\nd{dim}", fontsize=5, color="black")
        for edge in violations:
            src, dst = edge
            x0, y0 = _node_xy(src, topo.cols)
            x1, y1 = _node_xy(dst, topo.cols)
            ax.plot([x0, x1], [y0, y1], color="red", linewidth=3, alpha=0.75)
        ax.set_title(f"color {color}")

    for idx in range(plan.C, len(axes_list)):
        axes_list[idx].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _render_case(k: int, n: int, out_dir: Path) -> None:
    topo = TDMFlatButterfly(k=k, n=n, rows=8, cols=8)
    violations = _validate_color_plan(topo)
    if violations:
        print(f"[WARN] k={k}, n={n}, violations={len(violations)}")
    _plot_overview(topo, out_dir / f"k{k}_n{n}_overview.png", violations)
    _plot_colors(topo, out_dir / f"k{k}_n{n}_colors.png", violations)


def main() -> None:
    out_dir = Path("outputs/tdm_flatbf_8x8/coloring")
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, n in ((8, 2), (4, 3), (2, 6)):
        _render_case(k, n, out_dir)
    print(f"wrote coloring images to {out_dir}")


if __name__ == "__main__":
    main()

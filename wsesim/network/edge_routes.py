"""Precomputed per-edge routes for color NoC simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from wsesim.network.color import ColorPlan
from wsesim.network.color_routes import mesh_dims, xy_path


@dataclass(slots=True)
class EdgeRouteTable:
    """Maps (src, dst) to hop-by-hop path; color field for per-VN flow control."""

    paths: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    edge_colors: dict[tuple[int, int], int] = field(default_factory=dict)
    color_plan: ColorPlan | None = None

    def path_for(self, src: int, dst: int) -> list[int] | None:
        return self.paths.get((src, dst))

    def color_for(self, src: int, dst: int) -> int:
        return self.edge_colors.get((src, dst), 0)


def build_routes_for_traffic(
    traffic: list[dict],
    num_nodes: int,
    num_colors: int = 16,
    cols: int | None = None,
) -> tuple[EdgeRouteTable, list[dict]]:
    """Build per-edge XY paths; one virtual color per unique edge."""
    rows, cols = mesh_dims(num_nodes, cols)
    table = EdgeRouteTable()
    updated: list[dict] = []

    for pkt in traffic:
        src = int(pkt["src_core"])
        dst = int(pkt["dst_core"])
        key = (src, dst)
        if key not in table.paths:
            table.paths[key] = xy_path(src, dst, rows, cols)
            table.edge_colors[key] = len(table.edge_colors) % max(num_colors, len(table.paths))
        enriched = dict(pkt)
        enriched["color"] = table.edge_colors[key]
        enriched["seq"] = 0
        updated.append(enriched)

    nc = max(num_colors, len(table.edge_colors), 1)
    table.color_plan = ColorPlan.empty(num_nodes, nc)
    return table, updated

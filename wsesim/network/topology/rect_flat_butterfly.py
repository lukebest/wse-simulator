"""TDM flattened butterfly on a rectangular mesh (mixed-radix 2-flat).

Each row is a complete graph in dimension 0 (radix = cols); each column is a
complete graph in dimension 1 (radix = rows).  This is the natural extension
when rows*cols is not k^n.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wsesim.network.color_planners import ColorPlannerConfig, build_color_plan
from wsesim.network.tdm_coloring import ColorPlan
from wsesim.network.topology.base import Topology


@dataclass(slots=True)
class RectFlatButterfly(Topology):
    rows: int
    cols: int
    coloring_strategy: str = "greedy_first_fit"
    color_planner_config: ColorPlannerConfig | None = None
    _color_plan: ColorPlan | None = field(default=None, repr=False)
    _color_plan_override: ColorPlan | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("RectFlatButterfly requires positive rows/cols.")

    @property
    def n(self) -> int:
        return 2

    @property
    def k_dims(self) -> tuple[int, int]:
        return (self.cols, self.rows)

    def build(self, num_nodes: int) -> dict[int, list[int]]:
        expected = self.rows * self.cols
        if num_nodes != expected:
            raise ValueError("RectFlatButterfly num_nodes must equal rows*cols.")
        graph: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        for node in range(num_nodes):
            r, c = divmod(node, self.cols)
            if r > 0:
                graph[node].append((r - 1) * self.cols + c)
            if r < self.rows - 1:
                graph[node].append((r + 1) * self.cols + c)
            if c > 0:
                graph[node].append(r * self.cols + (c - 1))
            if c < self.cols - 1:
                graph[node].append(r * self.cols + (c + 1))
        return graph

    def to_coords(self, node_id: int) -> tuple[int, int]:
        r, c = divmod(node_id, self.cols)
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            raise ValueError("node_id out of range")
        return (c, r)

    def to_node(self, coords: tuple[int, ...]) -> int:
        if len(coords) != 2:
            raise ValueError("coords dimension mismatch")
        c, r = coords
        if c < 0 or c >= self.cols or r < 0 or r >= self.rows:
            raise ValueError("coords out of range")
        return r * self.cols + c

    def logical_neighbors(self, node: int) -> dict[int, list[int]]:
        r, c = divmod(node, self.cols)
        result: dict[int, list[int]] = {0: [], 1: []}
        for c2 in range(self.cols):
            if c2 != c:
                result[0].append(r * self.cols + c2)
        for r2 in range(self.rows):
            if r2 != r:
                result[1].append(r2 * self.cols + c)
        result[0] = sorted(result[0])
        result[1] = sorted(result[1])
        return result

    def logical_links(self) -> list[tuple[int, int, int]]:
        links: list[tuple[int, int, int]] = []
        for node in range(self.rows * self.cols):
            for dim, peers in self.logical_neighbors(node).items():
                for peer in peers:
                    links.append((node, peer, dim))
        return links

    def has_logical_link(self, src: int, dst: int) -> bool:
        r0, c0 = divmod(src, self.cols)
        r1, c1 = divmod(dst, self.cols)
        return r0 == r1 or c0 == c1

    def physical_path(self, src: int, dst: int) -> list[tuple[int, int]]:
        if src == dst:
            return []
        path: list[tuple[int, int]] = []
        sr, sc = divmod(src, self.cols)
        dr, dc = divmod(dst, self.cols)
        c = sc
        while c != dc:
            nc = c + (1 if dc > c else -1)
            path.append((sr * self.cols + c, sr * self.cols + nc))
            c = nc
        r = sr
        while r != dr:
            nr = r + (1 if dr > r else -1)
            path.append((r * self.cols + dc, nr * self.cols + dc))
            r = nr
        return path

    def physical_links(self) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for node in range(self.rows * self.cols):
            r, c = divmod(node, self.cols)
            if c + 1 < self.cols:
                right = r * self.cols + (c + 1)
                edges.append((node, right))
                edges.append((right, node))
            if r + 1 < self.rows:
                down = (r + 1) * self.cols + c
                edges.append((node, down))
                edges.append((down, node))
        return edges

    def dim_order_route(self, src: int, dst: int) -> list[tuple[int, int]]:
        if src == dst:
            return []
        hops: list[tuple[int, int]] = []
        cur = src
        cur_coords = list(self.to_coords(cur))
        dst_coords = self.to_coords(dst)
        for dim in range(self.n):
            if cur_coords[dim] == dst_coords[dim]:
                continue
            nxt_coords = list(cur_coords)
            nxt_coords[dim] = dst_coords[dim]
            nxt = self.to_node(tuple(nxt_coords))
            hops.append((cur, nxt))
            cur = nxt
            cur_coords = nxt_coords
        return hops

    def coloring(self) -> ColorPlan:
        if self._color_plan is None:
            if self._color_plan_override is not None:
                self._color_plan = self._color_plan_override
            else:
                config = self.color_planner_config
                if config is None:
                    config = ColorPlannerConfig(
                        planner=self.coloring_strategy,
                        topology_hint={
                            "k_dims": [self.cols, self.rows],
                            "rows": self.rows,
                            "cols": self.cols,
                        },
                    )
                self._color_plan, _ = build_color_plan(self, config)
        return self._color_plan

"""TDM flattened butterfly over a physical 2D mesh."""

from __future__ import annotations

from dataclasses import dataclass, field

from wsesim.network.color_planners import ColorPlannerConfig, build_color_plan
from wsesim.network.tdm_coloring import ColorPlan, min_colors_for_route, route_color
from wsesim.network.topology.base import Topology


@dataclass(slots=True)
class TDMFlatButterfly(Topology):
    k: int
    n: int
    rows: int
    cols: int
    coloring_strategy: str = "greedy_first_fit"
    color_planner_config: ColorPlannerConfig | None = None
    _color_plan: ColorPlan | None = field(default=None, repr=False)
    _color_plan_override: ColorPlan | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.k <= 1 or self.n <= 0:
            raise ValueError("TDMFlatButterfly requires k > 1 and n > 0.")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("TDMFlatButterfly requires positive rows/cols.")
        if self.k ** self.n != self.rows * self.cols:
            raise ValueError("TDMFlatButterfly requires k**n == rows*cols.")

    def build(self, num_nodes: int) -> dict[int, list[int]]:
        if num_nodes != self.rows * self.cols:
            raise ValueError("TDMFlatButterfly num_nodes must equal rows*cols.")
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

    def to_coords(self, node_id: int) -> tuple[int, ...]:
        if node_id < 0 or node_id >= self.rows * self.cols:
            raise ValueError("node_id out of range")
        value = node_id
        coords: list[int] = []
        for _ in range(self.n):
            coords.append(value % self.k)
            value //= self.k
        return tuple(coords)

    def to_node(self, coords: tuple[int, ...]) -> int:
        if len(coords) != self.n:
            raise ValueError("coords dimension mismatch")
        node = 0
        base = 1
        for axis in coords:
            if axis < 0 or axis >= self.k:
                raise ValueError("coords out of range")
            node += axis * base
            base *= self.k
        return node

    def logical_neighbors(self, node: int) -> dict[int, list[int]]:
        src = self.to_coords(node)
        result: dict[int, list[int]] = {}
        for dim in range(self.n):
            peers: list[int] = []
            for axis in range(self.k):
                if axis == src[dim]:
                    continue
                dst_coords = list(src)
                dst_coords[dim] = axis
                peers.append(self.to_node(tuple(dst_coords)))
            result[dim] = sorted(peers)
        return result

    def logical_links(self) -> list[tuple[int, int, int]]:
        links: list[tuple[int, int, int]] = []
        total = self.rows * self.cols
        for node in range(total):
            neighbors = self.logical_neighbors(node)
            for dim, peers in neighbors.items():
                for peer in peers:
                    links.append((node, peer, dim))
        return links

    def has_logical_link(self, src: int, dst: int) -> bool:
        src_coords = self.to_coords(src)
        dst_coords = self.to_coords(dst)
        diff_dims = [dim for dim in range(self.n) if src_coords[dim] != dst_coords[dim]]
        return len(diff_dims) == 1

    def physical_path(self, src: int, dst: int) -> list[tuple[int, int]]:
        if src == dst:
            return []
        path: list[tuple[int, int]] = []
        sr, sc = divmod(src, self.cols)
        dr, dc = divmod(dst, self.cols)
        c = sc
        while c != dc:
            nc = c + (1 if dc > c else -1)
            u = sr * self.cols + c
            v = sr * self.cols + nc
            path.append((u, v))
            c = nc
        r = sr
        while r != dr:
            nr = r + (1 if dr > r else -1)
            u = r * self.cols + dc
            v = nr * self.cols + dc
            path.append((u, v))
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
        """Dimension-order logical hops as directed (u, v) links."""
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
                        topology_hint={"k": self.k, "n": self.n},
                    )
                self._color_plan, _ = build_color_plan(self, config)
        return self._color_plan

    def route_color(self, src: int, dst: int) -> int | None:
        """Color for constant-color dim-order routing, if the default plan allows it."""
        return route_color(self.coloring(), self.dim_order_route(src, dst))

    def min_route_color(self, src: int, dst: int) -> tuple[int, int | None]:
        """Minimum C and color when this route alone is forced monochrome."""
        links = self.logical_links()
        paths = {(u, v): self.physical_path(u, v) for u, v, _ in links}
        return min_colors_for_route(
            logical_links=links,
            physical_paths=paths,
            physical_links=self.physical_links(),
            route_path=self.dim_order_route(src, dst),
        )

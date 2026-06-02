"""Restricted hypercube flattened butterfly carved from a larger parent mesh.

Keeps logical links from a parent ``TDMFlatButterfly(k=2, n=…)`` whose endpoints
and entire XY physical path stay inside a sub-rectangle of the parent grid.
Node IDs use the parent column stride (``parent_cols``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wsesim.network.tdm_coloring import ColorPlan, assign_colors
from wsesim.network.topology.base import Topology
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly


@dataclass(slots=True)
class RestrictedHypercubeFB(Topology):
    n: int
    parent_rows: int
    parent_cols: int
    keep_rows: int
    keep_cols: int
    k: int = 2
    _parent: TDMFlatButterfly | None = None
    _kept_logical: set[tuple[int, int]] = field(default_factory=set, init=False)
    _color_plan: ColorPlan | None = None

    def __post_init__(self) -> None:
        if self.k != 2:
            raise ValueError("RestrictedHypercubeFB currently supports k=2 only.")
        if self.keep_rows > self.parent_rows or self.keep_cols > self.parent_cols:
            raise ValueError("keep_rows/cols must fit inside parent mesh.")
        if 2 ** self.n != self.parent_rows * self.parent_cols:
            raise ValueError("parent mesh must satisfy k**n == parent_rows*parent_cols.")
        self._parent = TDMFlatButterfly(
            k=self.k, n=self.n, rows=self.parent_rows, cols=self.parent_cols
        )
        self._build_kept_links()

    @property
    def rows(self) -> int:
        return self.keep_rows

    @property
    def cols(self) -> int:
        return self.keep_cols

    def _in_sub(self, node: int) -> bool:
        r, c = divmod(node, self.parent_cols)
        return r < self.keep_rows and c < self.keep_cols

    def _build_kept_links(self) -> None:
        assert self._parent is not None
        kept: set[tuple[int, int]] = set()
        for u, v, _ in self._parent.logical_links():
            if not (self._in_sub(u) and self._in_sub(v)):
                continue
            path = self._parent.physical_path(u, v)
            if all(self._in_sub(a) and self._in_sub(b) for a, b in path):
                kept.add((u, v))
        self._kept_logical = kept

    def build(self, num_nodes: int) -> dict[int, list[int]]:
        expected = self.keep_rows * self.keep_cols
        if num_nodes != expected:
            raise ValueError("RestrictedHypercubeFB num_nodes must equal keep_rows*keep_cols.")
        graph: dict[int, list[int]] = {}
        for r in range(self.keep_rows):
            for c in range(self.keep_cols):
                node = r * self.parent_cols + c
                graph[node] = []
                if r > 0:
                    graph[node].append((r - 1) * self.parent_cols + c)
                if r < self.keep_rows - 1:
                    graph[node].append((r + 1) * self.parent_cols + c)
                if c > 0:
                    graph[node].append(r * self.parent_cols + (c - 1))
                if c < self.keep_cols - 1:
                    graph[node].append(r * self.parent_cols + (c + 1))
        return graph

    def to_coords(self, node_id: int) -> tuple[int, ...]:
        assert self._parent is not None
        return self._parent.to_coords(node_id)

    def to_node(self, coords: tuple[int, ...]) -> int:
        assert self._parent is not None
        return self._parent.to_node(coords)

    def logical_neighbors(self, node: int) -> dict[int, list[int]]:
        assert self._parent is not None
        full = self._parent.logical_neighbors(node)
        return {
            dim: sorted(peer for peer in peers if (node, peer) in self._kept_logical)
            for dim, peers in full.items()
        }

    def logical_links(self) -> list[tuple[int, int, int]]:
        assert self._parent is not None
        links: list[tuple[int, int, int]] = []
        for u, v, dim in self._parent.logical_links():
            if (u, v) in self._kept_logical:
                links.append((u, v, dim))
        return links

    def has_logical_link(self, src: int, dst: int) -> bool:
        return (src, dst) in self._kept_logical

    def physical_path(self, src: int, dst: int) -> list[tuple[int, int]]:
        assert self._parent is not None
        return self._parent.physical_path(src, dst)

    def physical_links(self) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for r in range(self.keep_rows):
            for c in range(self.keep_cols):
                node = r * self.parent_cols + c
                if c + 1 < self.keep_cols:
                    right = r * self.parent_cols + (c + 1)
                    edges.append((node, right))
                    edges.append((right, node))
                if r + 1 < self.keep_rows:
                    down = (r + 1) * self.parent_cols + c
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
            if self.has_logical_link(cur, nxt):
                hops.append((cur, nxt))
                cur = nxt
                cur_coords = nxt_coords
        return hops

    def coloring(self) -> ColorPlan:
        if self._color_plan is None:
            links = self.logical_links()
            paths = {(u, v): self.physical_path(u, v) for u, v, _ in links}
            self._color_plan = assign_colors(
                logical_links=links,
                physical_paths=paths,
                physical_links=self.physical_links(),
            )
        return self._color_plan

    def node_ids(self) -> list[int]:
        return [
            r * self.parent_cols + c
            for r in range(self.keep_rows)
            for c in range(self.keep_cols)
        ]

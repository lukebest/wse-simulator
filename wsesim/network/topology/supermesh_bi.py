"""SuperMeshBi topology: mesh with full peripheral enhanced links."""

from __future__ import annotations

from math import isqrt

from wsesim.network.topology.base import Topology
from wsesim.network.topology.mesh2d import Mesh2D


class SuperMeshBi(Topology):
    def __init__(self, rows: int | None = None, cols: int | None = None) -> None:
        self.rows = rows
        self.cols = cols

    def build(self, num_nodes: int) -> dict[int, list[int]]:
        rows, cols = self._resolve_shape(num_nodes)
        return Mesh2D(rows=rows, cols=cols).build(num_nodes)

    def enhanced_edges(self, num_nodes: int) -> set[tuple[int, int]]:
        rows, cols = self._resolve_shape(num_nodes)
        edges: set[tuple[int, int]] = set()

        # Top and bottom perimeter edges.
        for c in range(cols - 1):
            edges.add((0 * cols + c, 0 * cols + c + 1))
            edges.add(((rows - 1) * cols + c, (rows - 1) * cols + c + 1))

        # Left and right perimeter edges.
        for r in range(rows - 1):
            edges.add((r * cols + 0, (r + 1) * cols + 0))
            edges.add((r * cols + (cols - 1), (r + 1) * cols + (cols - 1)))

        return edges

    def _resolve_shape(self, num_nodes: int) -> tuple[int, int]:
        if self.rows is not None and self.cols is not None:
            if self.rows * self.cols != num_nodes:
                raise ValueError("SuperMeshBi rows*cols must equal num_nodes.")
            return self.rows, self.cols

        side = isqrt(num_nodes)
        if side * side != num_nodes:
            raise ValueError("SuperMeshBi requires explicit rows/cols for non-square node count.")
        return side, side

"""2D mesh topology."""

from __future__ import annotations

from math import isqrt

from wsesim.network.topology.base import Topology


class Mesh2D(Topology):
    def __init__(self, rows: int | None = None, cols: int | None = None) -> None:
        self.rows = rows
        self.cols = cols

    def build(self, num_nodes: int) -> dict[int, list[int]]:
        if self.rows is not None and self.cols is not None:
            rows = self.rows
            cols = self.cols
            if rows * cols != num_nodes:
                raise ValueError("Mesh2D rows*cols must equal num_nodes.")
        else:
            side = isqrt(num_nodes)
            if side * side != num_nodes:
                raise ValueError("Mesh2D requires a square node count.")
            rows = side
            cols = side

        graph: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        for node in range(num_nodes):
            r, c = divmod(node, cols)
            if r > 0:
                graph[node].append((r - 1) * cols + c)
            if r < rows - 1:
                graph[node].append((r + 1) * cols + c)
            if c > 0:
                graph[node].append(r * cols + (c - 1))
            if c < cols - 1:
                graph[node].append(r * cols + (c + 1))
        return graph

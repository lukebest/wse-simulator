"""Traditional butterfly-style topology mapped on a 2D plane."""

from __future__ import annotations

from math import isqrt

from wsesim.network.topology.base import Topology


class Butterfly(Topology):
    def __init__(self, rows: int | None = None, cols: int | None = None) -> None:
        self.rows = rows
        self.cols = cols

    def build(self, num_nodes: int) -> dict[int, list[int]]:
        rows, cols = self._resolve_shape(num_nodes)
        graph_sets: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
        bit_span = max(1, max(rows - 1, 1).bit_length())

        for c in range(cols - 1):
            step = 1 << (c % bit_span)
            for r in range(rows):
                src = r * cols + c
                if src >= num_nodes:
                    continue

                dst_straight = r * cols + (c + 1)
                if dst_straight < num_nodes:
                    graph_sets[src].add(dst_straight)
                    graph_sets[dst_straight].add(src)

                partner_row = (r ^ step) % rows
                dst_exchange = partner_row * cols + (c + 1)
                if dst_exchange < num_nodes:
                    graph_sets[src].add(dst_exchange)
                    graph_sets[dst_exchange].add(src)

        return {node: sorted(neighbors) for node, neighbors in graph_sets.items()}

    def _resolve_shape(self, num_nodes: int) -> tuple[int, int]:
        if self.rows is not None and self.cols is not None:
            if self.rows * self.cols != num_nodes:
                raise ValueError("Butterfly rows*cols must equal num_nodes.")
            return self.rows, self.cols

        side = isqrt(num_nodes)
        if side > 1 and side * side == num_nodes:
            return side, side
        return 1, num_nodes

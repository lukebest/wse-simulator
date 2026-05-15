"""2D torus topology."""

from __future__ import annotations

from math import isqrt

from wsesim.network.topology.base import Topology


class Torus2D(Topology):
    def build(self, num_nodes: int) -> dict[int, list[int]]:
        side = isqrt(num_nodes)
        if side * side != num_nodes:
            raise ValueError("Torus2D requires a square node count.")

        graph: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        for node in range(num_nodes):
            r, c = divmod(node, side)
            up = ((r - 1) % side) * side + c
            down = ((r + 1) % side) * side + c
            left = r * side + ((c - 1) % side)
            right = r * side + ((c + 1) % side)
            graph[node].extend([up, down, left, right])
        return graph

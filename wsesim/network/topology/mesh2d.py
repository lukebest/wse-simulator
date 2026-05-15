"""2D mesh topology."""

from __future__ import annotations

from math import isqrt

from wsesim.network.topology.base import Topology


class Mesh2D(Topology):
    def build(self, num_nodes: int) -> dict[int, list[int]]:
        side = isqrt(num_nodes)
        if side * side != num_nodes:
            raise ValueError("Mesh2D requires a square node count.")

        graph: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        for node in range(num_nodes):
            r, c = divmod(node, side)
            if r > 0:
                graph[node].append((r - 1) * side + c)
            if r < side - 1:
                graph[node].append((r + 1) * side + c)
            if c > 0:
                graph[node].append(r * side + (c - 1))
            if c < side - 1:
                graph[node].append(r * side + (c + 1))
        return graph

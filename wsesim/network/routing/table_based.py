"""Defect-aware precomputed table routing."""

from __future__ import annotations

from collections import deque

from wsesim.network.routing.base import RoutingAlgorithm


class TableBasedRouting(RoutingAlgorithm):
    def __init__(self, graph: dict[int, list[int]]) -> None:
        self._table: dict[int, dict[int, int]] = {}
        self._build(graph)

    def _build(self, graph: dict[int, list[int]]) -> None:
        for src in graph:
            self._table[src] = {}
            for dst in graph:
                if src == dst:
                    self._table[src][dst] = dst
                else:
                    self._table[src][dst] = _shortest_next_hop(src, dst, graph)

    def next_hop(self, current: int, dst: int, graph: dict[int, list[int]]) -> int:
        if current not in self._table or dst not in self._table[current]:
            raise ValueError(f"No route in table for {current}->{dst}.")
        return self._table[current][dst]


def _shortest_next_hop(src: int, dst: int, graph: dict[int, list[int]]) -> int:
    queue = deque([src])
    parent = {src: -1}
    while queue:
        node = queue.popleft()
        if node == dst:
            break
        for neighbor in graph.get(node, []):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if dst not in parent:
        raise ValueError(f"No path from {src} to {dst}.")
    hop = dst
    while parent[hop] != src and parent[hop] != -1:
        hop = parent[hop]
    return hop

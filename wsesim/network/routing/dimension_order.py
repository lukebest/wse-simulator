"""Dimension-order (XY-like) routing over adjacency graph."""

from __future__ import annotations

from collections import deque

from wsesim.network.routing.base import RoutingAlgorithm


class DimensionOrderRouting(RoutingAlgorithm):
    def next_hop(self, current: int, dst: int, graph: dict[int, list[int]]) -> int:
        if current == dst:
            return dst
        return _bfs_next_hop(current, dst, graph)


def _bfs_next_hop(current: int, dst: int, graph: dict[int, list[int]]) -> int:
    queue = deque([current])
    parent = {current: -1}
    while queue:
        node = queue.popleft()
        if node == dst:
            break
        for neighbor in graph.get(node, []):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if dst not in parent:
        raise ValueError(f"No path from {current} to {dst}.")

    hop = dst
    while parent[hop] != current and parent[hop] != -1:
        hop = parent[hop]
    return hop

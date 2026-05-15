"""UGAL-like adaptive routing (simplified)."""

from __future__ import annotations

import random

from wsesim.network.routing.base import RoutingAlgorithm
from wsesim.network.routing.dimension_order import _bfs_next_hop


class UGALRouting(RoutingAlgorithm):
    def __init__(self, seed: int = 1234) -> None:
        self._rng = random.Random(seed)

    def next_hop(self, current: int, dst: int, graph: dict[int, list[int]]) -> int:
        if current == dst:
            return dst
        if self._rng.random() < 0.5:
            return _bfs_next_hop(current, dst, graph)

        intermediate = self._rng.choice([n for n in graph if n not in (current, dst)])
        try:
            return _bfs_next_hop(current, intermediate, graph)
        except ValueError:
            return _bfs_next_hop(current, dst, graph)

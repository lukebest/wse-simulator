"""Routing interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RoutingAlgorithm(ABC):
    @abstractmethod
    def next_hop(self, current: int, dst: int, graph: dict[int, list[int]]) -> int:
        raise NotImplementedError

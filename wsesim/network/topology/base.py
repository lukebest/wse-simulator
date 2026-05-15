"""Topology interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Topology(ABC):
    @abstractmethod
    def build(self, num_nodes: int) -> dict[int, list[int]]:
        raise NotImplementedError

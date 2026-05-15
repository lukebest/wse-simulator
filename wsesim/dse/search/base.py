"""Search strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from wsesim.core.config import WSEConfig


class SearchStrategy(ABC):
    @abstractmethod
    def suggest(self, history: list[tuple[WSEConfig, float]]) -> WSEConfig:
        raise NotImplementedError

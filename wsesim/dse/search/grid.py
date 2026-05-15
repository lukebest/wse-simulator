"""Grid search strategy."""

from __future__ import annotations

from copy import deepcopy

from wsesim.core.config import WSEConfig
from wsesim.dse.search.base import SearchStrategy


class GridSearch(SearchStrategy):
    def __init__(self, candidates: list[WSEConfig]) -> None:
        if not candidates:
            raise ValueError("GridSearch requires at least one candidate.")
        self._candidates = candidates

    def suggest(self, history: list[tuple[WSEConfig, float]]) -> WSEConfig:
        idx = min(len(history), len(self._candidates) - 1)
        return deepcopy(self._candidates[idx])

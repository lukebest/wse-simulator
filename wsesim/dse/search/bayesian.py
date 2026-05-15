"""Bayesian search placeholder."""

from __future__ import annotations

from copy import deepcopy

from wsesim.core.config import WSEConfig
from wsesim.dse.search.base import SearchStrategy


class BayesianSearch(SearchStrategy):
    def __init__(self, base: WSEConfig) -> None:
        self.base = base

    def suggest(self, history: list[tuple[WSEConfig, float]]) -> WSEConfig:
        # Placeholder fallback until skopt integration is added.
        cfg = deepcopy(self.base)
        if history:
            cfg.compute.pe_width = min(64, max(8, cfg.compute.pe_width + 8))
        return cfg

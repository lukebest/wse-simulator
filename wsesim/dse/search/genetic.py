"""Genetic search placeholder."""

from __future__ import annotations

from copy import deepcopy

from wsesim.core.config import WSEConfig
from wsesim.dse.search.base import SearchStrategy


class GeneticSearch(SearchStrategy):
    def __init__(self, base: WSEConfig) -> None:
        self.base = base

    def suggest(self, history: list[tuple[WSEConfig, float]]) -> WSEConfig:
        cfg = deepcopy(self.base)
        if history and len(history) % 2 == 0:
            cfg.network.noc.link_bw_flits_per_cycle += 1
        return cfg

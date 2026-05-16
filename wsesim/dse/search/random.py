"""Random search strategy."""

from __future__ import annotations

from copy import deepcopy
import random

from wsesim.core.config import WSEConfig
from wsesim.dse.search.base import SearchStrategy


class RandomSearch(SearchStrategy):
    def __init__(self, base: WSEConfig, seed: int = 1234) -> None:
        self.base = base
        self._rng = random.Random(seed)

    def suggest(self, history: list[tuple[WSEConfig, float]]) -> WSEConfig:
        cfg = deepcopy(self.base)
        cfg.compute.pe_width = self._rng.choice([8, 16, 32])
        cfg.network.noc.buffer_depth = self._rng.choice([4, 8, 16])
        cfg.network.noc.num_vcs = self._rng.choice([1, 2, 4])
        cfg.network.noc.link_bw_flits_per_cycle = self._rng.choice([1, 2, 4])

        max_gateways = max(1, min(cfg.wafer.cores_per_reticle, 8))
        gateway_candidates = sorted({1, 2, 4, max_gateways})
        cfg.network.gateways_per_reticle = self._rng.choice(gateway_candidates)
        cfg.network.gateway_policy = self._rng.choice(["nearest", "load_aware"])
        return cfg

"""DSE engine orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult
from wsesim.dse.search.base import SearchStrategy


Evaluator = Callable[[WSEConfig], SimResult]


@dataclass(slots=True)
class DSEEngine:
    base_config: WSEConfig
    strategy: SearchStrategy
    evaluator: Evaluator
    workers: int = 1

    def run(self, trials: int) -> list[tuple[WSEConfig, float]]:
        history: list[tuple[WSEConfig, float]] = []
        if self.workers <= 1:
            for _ in range(trials):
                cfg = self.strategy.suggest(history)
                result = self.evaluator(deepcopy(cfg))
                score = result.dse_score(self.base_config.dse.score_weights)
                history.append((cfg, score))
            return history

        candidates = [self.strategy.suggest(history) for _ in range(trials)]
        with Pool(self.workers) as pool:
            results = pool.map(self.evaluator, [deepcopy(c) for c in candidates])
        for cfg, result in zip(candidates, results, strict=True):
            history.append((cfg, result.dse_score(self.base_config.dse.score_weights)))
        return history

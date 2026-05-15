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
class DSETrial:
    config: WSEConfig
    result: SimResult
    score: float


@dataclass(slots=True)
class DSEEngine:
    base_config: WSEConfig
    strategy: SearchStrategy
    evaluator: Evaluator
    workers: int = 1

    def run_detailed(self, trials: int) -> list[DSETrial]:
        trial_history: list[DSETrial] = []
        score_history: list[tuple[WSEConfig, float]] = []
        if self.workers <= 1:
            for _ in range(trials):
                cfg = self.strategy.suggest(score_history)
                result = self.evaluator(deepcopy(cfg))
                score = result.dse_score(self.base_config.dse.score_weights)
                trial = DSETrial(config=cfg, result=result, score=score)
                trial_history.append(trial)
                score_history.append((cfg, score))
            return trial_history

        candidates = [self.strategy.suggest(score_history) for _ in range(trials)]
        with Pool(self.workers) as pool:
            results = pool.map(self.evaluator, [deepcopy(c) for c in candidates])
        for cfg, result in zip(candidates, results, strict=True):
            score = result.dse_score(self.base_config.dse.score_weights)
            trial = DSETrial(config=cfg, result=result, score=score)
            trial_history.append(trial)
            score_history.append((cfg, score))
        return trial_history

    def run(self, trials: int) -> list[tuple[WSEConfig, float]]:
        return [(trial.config, trial.score) for trial in self.run_detailed(trials)]

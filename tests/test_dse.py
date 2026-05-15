from __future__ import annotations

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult
from wsesim.dse.engine import DSEEngine
from wsesim.dse.report import summarize_best
from wsesim.dse.search.random import RandomSearch


def _evaluate(cfg: WSEConfig) -> SimResult:
    return SimResult(total_latency_cycles=max(1, 1000 // cfg.compute.pe_width))


def test_dse_runs_and_reports_best() -> None:
    base = WSEConfig()
    dse = DSEEngine(base_config=base, strategy=RandomSearch(base, seed=1), evaluator=_evaluate)
    history = dse.run(trials=4)
    summary = summarize_best(history)
    assert len(history) == 4
    assert summary["best_score"] is not None

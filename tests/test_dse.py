from __future__ import annotations

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult
from wsesim.dse.engine import DSEEngine
from wsesim.dse.report import summarize_best
from wsesim.dse.search.random import RandomSearch


def _evaluate(cfg: WSEConfig) -> SimResult:
    congestion_scale = max(1, 64 // cfg.compute.pe_width)
    return SimResult(
        total_latency_cycles=max(1, 1000 // cfg.compute.pe_width),
        vc_wait_cycles=5 * congestion_scale,
        buffer_wait_cycles=3 * congestion_scale,
        link_wait_cycles=2 * congestion_scale,
    )


def test_dse_runs_and_reports_best() -> None:
    base = WSEConfig()
    dse = DSEEngine(base_config=base, strategy=RandomSearch(base, seed=1), evaluator=_evaluate)
    history = dse.run(trials=4)
    summary = summarize_best(history)
    assert len(history) == 4
    assert summary["best_score"] is not None


def test_dse_score_uses_congestion_penalties() -> None:
    weights = {
        "total_latency_cycles": -1.0,
        "vc_wait_cycles": -2.0,
        "buffer_wait_cycles": -1.0,
        "link_wait_cycles": -1.0,
    }
    low_congestion = SimResult(
        total_latency_cycles=100,
        vc_wait_cycles=1,
        buffer_wait_cycles=1,
        link_wait_cycles=1,
    )
    high_congestion = SimResult(
        total_latency_cycles=100,
        vc_wait_cycles=20,
        buffer_wait_cycles=20,
        link_wait_cycles=20,
    )
    assert low_congestion.dse_score(weights) > high_congestion.dse_score(weights)

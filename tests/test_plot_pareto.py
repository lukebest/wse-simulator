from __future__ import annotations

from pathlib import Path

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult
from wsesim.dse.engine import DSEEngine
from wsesim.dse.plot import plot_pareto
from wsesim.dse.report import export_pareto_csv, export_trials_csv, pareto_front
from wsesim.dse.search.random import RandomSearch


def _evaluate(cfg: WSEConfig) -> SimResult:
    congestion_scale = max(1, 64 // cfg.compute.pe_width)
    return SimResult(
        total_latency_cycles=max(1, 1000 // cfg.compute.pe_width),
        vc_wait_cycles=5 * congestion_scale,
        buffer_wait_cycles=3 * congestion_scale,
        link_wait_cycles=2 * congestion_scale,
        network_throughput=float(cfg.compute.pe_width) / 64.0,
    )


def test_plot_pareto_generates_png_files(tmp_path: Path) -> None:
    base = WSEConfig()
    dse = DSEEngine(base_config=base, strategy=RandomSearch(base, seed=7), evaluator=_evaluate)
    trials = dse.run_detailed(trials=6)
    front = pareto_front(trials)

    trials_csv = export_trials_csv(trials, tmp_path / "dse_trials.csv")
    pareto_csv = export_pareto_csv(front, tmp_path / "dse_pareto.csv")
    out1, out2 = plot_pareto(trials_csv=trials_csv, pareto_csv=pareto_csv, output_dir=tmp_path)

    assert out1.exists()
    assert out2.exists()
    assert out1.stat().st_size > 0
    assert out2.stat().st_size > 0

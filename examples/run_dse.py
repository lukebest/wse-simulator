"""Run a tiny DSE loop with random search."""

from __future__ import annotations

from pathlib import Path

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult
from wsesim.dse.engine import DSEEngine
from wsesim.dse.report import (
    export_pareto_csv,
    export_trials_csv,
    export_trials_json,
    pareto_front,
    summarize_best_trial,
)
from wsesim.dse.search.random import RandomSearch


def evaluate(config: WSEConfig) -> SimResult:
    # Placeholder evaluator: lower pe_width gives higher latency and more congestion.
    base_latency = max(1, 10_000 // config.compute.pe_width)
    congestion_scale = max(1, 64 // config.compute.pe_width)
    result = SimResult(
        total_latency_cycles=base_latency,
        vc_wait_cycles=5 * congestion_scale,
        buffer_wait_cycles=4 * congestion_scale,
        link_wait_cycles=3 * congestion_scale,
        network_throughput=float(config.compute.pe_width) / 64.0,
    )
    return result


def main() -> None:
    base = WSEConfig()
    engine = DSEEngine(
        base_config=base,
        strategy=RandomSearch(base),
        evaluator=evaluate,
        workers=1,
    )
    trials = engine.run_detailed(trials=8)
    print("best:", summarize_best_trial(trials))
    front = pareto_front(trials)
    print("pareto_size:", len(front))
    for idx, trial in enumerate(front, start=1):
        print(
            f"pareto[{idx}] score={trial.score:.3f} "
            f"lat={trial.result.total_latency_cycles} "
            f"vc_wait={trial.result.vc_wait_cycles} "
            f"buf_wait={trial.result.buffer_wait_cycles} "
            f"link_wait={trial.result.link_wait_cycles} "
            f"thr={trial.result.network_throughput:.3f}"
        )

    output_dir = Path("outputs")
    trials_json = export_trials_json(trials, output_dir / "dse_trials.json")
    trials_csv = export_trials_csv(trials, output_dir / "dse_trials.csv")
    pareto_csv = export_pareto_csv(front, output_dir / "dse_pareto.csv")
    print("exported:", trials_json, trials_csv, pareto_csv)


if __name__ == "__main__":
    main()

"""Run a tiny DSE loop with random search."""

from __future__ import annotations

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult
from wsesim.dse.engine import DSEEngine
from wsesim.dse.report import summarize_best
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
    history = engine.run(trials=5)
    print(summarize_best(history))


if __name__ == "__main__":
    main()

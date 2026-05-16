"""Run a tiny DSE loop with random search."""

from __future__ import annotations

from pathlib import Path

from wsesim.core.config import WSEConfig
from wsesim.dse.engine import DSEEngine
from wsesim.dse.evaluator_deepseek import evaluate_deepseek_v4_pro_ffn
from wsesim.dse.report import (
    export_pareto_csv,
    export_trials_csv,
    export_trials_json,
    pareto_front,
    summarize_best_trial,
)
from wsesim.dse.search.random import RandomSearch


def main() -> None:
    base = WSEConfig()
    base.wafer.reticle_rows = 6
    base.wafer.reticle_cols = 8
    base.wafer.reticle_dead_positions = ((1, 0), (2, 0), (3, 0), (4, 0))
    base.compute.pe_type = "cube"
    base.compute.pe_freq_ghz = 2.0
    base.compute.l1_capacity_kb = 2048
    base.compute.cube_m_tile = 4
    base.compute.cube_k_tile = 32
    base.compute.cube_n_tile = 16
    base.compute.cube_startup_cycles = 27
    base.compute.cube_steady_cycles = 5
    base.memory.per_core_bandwidth_gbps = 256.0
    base.memory.per_core_latency_ns = 100.0
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.hidden_dim = 7168
    base.workload.expert_ffn_dim = 3072
    base.workload.num_routed_experts = 384
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.decode_tokens = 32
    base.workload.routing_skew_alpha = 1.2
    base.workload.capacity_factor = 1.25
    base.workload.mapping_strategy = "expert_affinity"
    base.network.gateways_per_reticle = 4
    base.network.gateway_policy = "load_aware"

    engine = DSEEngine(
        base_config=base,
        strategy=RandomSearch(base),
        evaluator=evaluate_deepseek_v4_pro_ffn,
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
            f"gw_hops={trial.result.gateway_noc_hops} "
            f"gw_peak={trial.result.gateway_peak_load} "
            f"thr={trial.result.network_throughput:.3f}"
        )

    output_dir = Path("outputs")
    trials_json = export_trials_json(trials, output_dir / "dse_trials.json")
    trials_csv = export_trials_csv(trials, output_dir / "dse_trials.csv")
    pareto_csv = export_pareto_csv(front, output_dir / "dse_pareto.csv")
    print("exported:", trials_json, trials_csv, pareto_csv)


if __name__ == "__main__":
    main()

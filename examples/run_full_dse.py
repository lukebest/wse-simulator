"""Run a comprehensive DSE loop for DeepSeek-V4-Pro FFN."""

from __future__ import annotations

import argparse
from pathlib import Path

from wsesim.core.config import WSEConfig
from wsesim.dse.engine import DSEEngine
from wsesim.dse.evaluator_deepseek import evaluate_deepseek_v4_pro_ffn
from wsesim.dse.pipeline_analysis import (
    compute_overlap_breakdown,
    compute_pipeline_breakdown,
    format_overlap_breakdown_markdown,
)
from wsesim.dse.plot import (
    plot_bandwidth_utilization,
    plot_latency_breakdown,
    plot_pipeline_gantt,
)
from wsesim.dse.report import (
    export_pareto_csv,
    export_trials_csv,
    export_trials_json,
    pareto_front,
    summarize_best_trial,
)
from wsesim.dse.search.random import RandomSearch


def _bound_kind(trial) -> str:
    compute = trial.result.compute_cycles
    memory = trial.result.memory_stall_cycles
    network = trial.result.network_cycles + trial.result.io_injection_cycles + trial.result.allreduce_cycles
    if memory >= max(compute, network):
        return "memory-bound"
    if compute >= max(memory, network):
        return "compute-bound"
    return "network-bound"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comprehensive DeepSeek-V4-Pro FFN DSE.")
    parser.add_argument("--trials", type=int, default=50, help="Number of DSE trials.")
    parser.add_argument("--workers", type=int, default=1, help="Number of evaluator workers.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for search.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate pipeline visualization charts for top Pareto trials.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write DSE artifacts.",
    )
    parser.add_argument(
        "--partition-strategies",
        nargs="+",
        default=None,
        help="Restrict partition search (e.g. col entwined_ring for N-split only).",
    )
    parser.add_argument(
        "--breakdown-best",
        action="store_true",
        help="Write max-path overlap latency breakdown for the best trial.",
    )
    args = parser.parse_args()

    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.hidden_dim = 7168
    base.workload.expert_ffn_dim = 3072
    base.workload.num_routed_experts = 384
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.mapping_strategy = "expert_affinity"

    engine = DSEEngine(
        base_config=base,
        strategy=RandomSearch(
            base,
            seed=args.seed,
            partition_strategies=args.partition_strategies,
        ),
        evaluator=evaluate_deepseek_v4_pro_ffn,
        workers=max(1, args.workers),
    )
    trials = engine.run_detailed(trials=max(1, args.trials))
    best = summarize_best_trial(trials)
    print("best:", best)

    best_trial = max(trials, key=lambda trial: trial.score)
    if args.breakdown_best:
        breakdown = compute_overlap_breakdown(best_trial.config)
        label = (
            f"{best_trial.config.workload.partition_strategy}/"
            f"s{best_trial.config.workload.partition_shards}/"
            f"b{best_trial.config.workload.decode_tokens}/"
            f"{best_trial.config.network.noc.topology}:{best_trial.config.network.now.topology}"
        )
        md = format_overlap_breakdown_markdown(breakdown, title=f"Best Trial Latency Breakdown — {label}")
        breakdown_path = args.output_dir / "best_trial_latency_breakdown.md"
        breakdown_path.parent.mkdir(parents=True, exist_ok=True)
        breakdown_path.write_text(md, encoding="utf-8")
        print("breakdown:", breakdown_path)
        print(md)

    ranked = sorted(trials, key=lambda trial: trial.score, reverse=True)[:5]
    for idx, trial in enumerate(ranked, start=1):
        result = trial.result
        cfg = trial.config
        print(
            f"top[{idx}] score={trial.score:.3f} "
            f"lat={result.total_latency_cycles} "
            f"compute={result.compute_cycles} mem={result.memory_stall_cycles} "
            f"net={result.network_cycles} io={result.io_injection_cycles} allr={result.allreduce_cycles} "
            f"batch={cfg.workload.decode_tokens} part={cfg.workload.partition_strategy}/{cfg.workload.partition_shards} "
            f"noc={cfg.network.noc.topology}:{cfg.network.noc.routing}:{cfg.network.noc.flow_control} "
            f"now={cfg.network.now.topology}:{cfg.network.now.routing}:{cfg.network.now.flow_control} "
            f"bound={_bound_kind(trial)}"
        )

    front = pareto_front(trials)
    print("pareto_size:", len(front))

    output_dir = args.output_dir
    trials_json = export_trials_json(trials, output_dir / "full_dse_trials.json")
    trials_csv = export_trials_csv(trials, output_dir / "full_dse_trials.csv")
    pareto_csv = export_pareto_csv(front, output_dir / "full_dse_pareto.csv")
    print("exported:", trials_json, trials_csv, pareto_csv)

    if args.visualize and front:
        top_front = sorted(front, key=lambda trial: trial.score, reverse=True)[:5]
        breakdowns = []
        for trial in top_front:
            cfg = trial.config
            label = (
                f"{cfg.workload.partition_strategy}/s{cfg.workload.partition_shards}/b{cfg.workload.decode_tokens}/"
                f"{cfg.network.noc.topology}:{cfg.network.now.topology}"
            )
            breakdowns.append(compute_pipeline_breakdown(cfg, config_label=label))
        out_gantt = plot_pipeline_gantt(breakdowns, output_dir / "pareto_pipeline_gantt.png")
        out_breakdown = plot_latency_breakdown(
            breakdowns, output_dir / "pareto_pipeline_latency_breakdown.png"
        )
        out_bw = plot_bandwidth_utilization(
            breakdowns, output_dir / "pareto_pipeline_bw_utilization.png"
        )
        print("visualized:", out_gantt, out_breakdown, out_bw)


if __name__ == "__main__":
    main()

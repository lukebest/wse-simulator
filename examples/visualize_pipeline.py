"""Visualize DeepSeek-V4-Pro FFN pipeline breakdowns across DSE choices."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from wsesim.core.config import WSEConfig
from wsesim.dse.pipeline_analysis import PipelineBreakdown, compute_pipeline_breakdown
from wsesim.dse.plot import (
    plot_bandwidth_utilization,
    plot_latency_breakdown,
    plot_pipeline_gantt,
)


def _parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _parse_str_csv(raw: str) -> list[str]:
    values: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(token)
    return values


def _iter_configs(base: WSEConfig, strategies: list[str], batch_sizes: list[int], shards: list[int]):
    for strategy in strategies:
        for batch_size in batch_sizes:
            for shard in shards:
                if strategy == "expert" and shard != 1:
                    continue
                cfg = deepcopy(base)
                cfg.workload.partition_strategy = strategy
                cfg.workload.partition_shards = shard
                cfg.workload.decode_tokens = batch_size
                yield cfg


def _label(cfg: WSEConfig) -> str:
    return (
        f"{cfg.workload.partition_strategy}/s{cfg.workload.partition_shards}/b{cfg.workload.decode_tokens}/"
        f"{cfg.network.noc.topology}:{cfg.network.now.topology}"
    )


def generate_pipeline_visuals(
    breakdowns: list[PipelineBreakdown],
    output_dir: Path,
    prefix: str = "pipeline",
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gantt_path = plot_pipeline_gantt(breakdowns, output_dir / f"{prefix}_gantt.png")
    breakdown_path = plot_latency_breakdown(breakdowns, output_dir / f"{prefix}_latency_breakdown.png")
    bw_path = plot_bandwidth_utilization(breakdowns, output_dir / f"{prefix}_bw_utilization.png")
    return gantt_path, breakdown_path, bw_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DSE pipeline visualizations.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to write charts.")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="4,16",
        help="Comma-separated decode token counts.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="expert,col,k_split",
        help="Comma-separated partition strategies.",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default="1,4",
        help="Comma-separated shard counts.",
    )
    args = parser.parse_args()

    batch_sizes = _parse_int_csv(args.batch_sizes)
    strategies = _parse_str_csv(args.strategies)
    shards = _parse_int_csv(args.shards)

    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.hidden_dim = 7168
    base.workload.expert_ffn_dim = 3072
    base.workload.num_routed_experts = 384
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.mapping_strategy = "expert_affinity"

    breakdowns = [
        compute_pipeline_breakdown(config=cfg, config_label=_label(cfg))
        for cfg in _iter_configs(base, strategies=strategies, batch_sizes=batch_sizes, shards=shards)
    ]
    outputs = generate_pipeline_visuals(breakdowns, args.output_dir, prefix="pipeline")
    print("generated:", *outputs)


if __name__ == "__main__":
    main()

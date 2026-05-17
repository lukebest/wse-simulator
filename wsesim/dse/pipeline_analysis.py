"""Pipeline-oriented analysis for DeepSeek-V4-Pro FFN DSE results."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from wsesim.core.config import WSEConfig
from wsesim.dse.evaluator_deepseek import (
    _effective_partition_shards,
    _estimate_allreduce_cycles,
    _estimate_network_metrics,
)
from wsesim.workload.generator import (
    DeepSeekV4ProFFNProfile,
    generate_deepseek_v4_pro_decode_ffn_workload,
)


@dataclass(slots=True)
class PipelineStage:
    name: str
    category: str
    start_cycle: int
    duration_cycles: int

    @property
    def end_cycle(self) -> int:
        return self.start_cycle + self.duration_cycles


@dataclass(slots=True)
class PipelineBreakdown:
    config_label: str
    stages: list[PipelineStage]
    total_cycles: int
    partition_strategy: str
    partition_shards: int
    batch_size: int
    peak_mem_bw_utilization: float

    def category_cycles(self) -> dict[str, int]:
        totals = {"compute": 0, "memory": 0, "network": 0, "io": 0, "allreduce": 0}
        for stage in self.stages:
            if stage.category in totals:
                totals[stage.category] += stage.duration_cycles
        return totals


def compute_pipeline_breakdown(config: WSEConfig, config_label: str | None = None) -> PipelineBreakdown:
    """Create a sequential pipeline view aligned with DeepSeek evaluator formulas."""
    profile = DeepSeekV4ProFFNProfile(
        hidden_dim=config.workload.hidden_dim,
        expert_ffn_dim=config.workload.expert_ffn_dim,
        num_routed_experts=config.workload.num_routed_experts,
        num_shared_experts=config.workload.num_shared_experts,
        top_k=config.workload.top_k,
        decode_tokens=config.workload.decode_tokens,
        routing_skew_alpha=config.workload.routing_skew_alpha,
        capacity_factor=config.workload.capacity_factor,
    )
    workload = generate_deepseek_v4_pro_decode_ffn_workload(profile)
    shards = _effective_partition_shards(workload, config)

    hidden = int(workload.metadata["hidden_dim"])
    ffn_dim = int(workload.metadata["expert_ffn_dim"])
    decode_tokens = int(workload.metadata["decode_tokens"])
    active_routed = int(workload.metadata["active_routed_experts"])
    shared_experts = int(workload.metadata["num_shared_experts"])
    active_experts = active_routed + shared_experts
    fp_bytes = 2

    mem_bw_bytes_per_cycle = _memory_bw_bytes_per_cycle(config)
    io_bw_bytes_per_cycle = max(1.0, config.wafer.io_bandwidth_gbps / max(config.network.noc.freq_ghz, 0.1))

    w1_weight_bytes = active_experts * hidden * ffn_dim * fp_bytes
    w3_weight_bytes = active_experts * hidden * ffn_dim * fp_bytes
    w2_weight_bytes = active_experts * hidden * ffn_dim * fp_bytes

    if config.workload.partition_strategy == "expert":
        memory_divisor = 1
    else:
        memory_divisor = max(1, shards)

    mem_w1_cycles = int(w1_weight_bytes / memory_divisor / mem_bw_bytes_per_cycle)
    mem_w3_cycles = int(w3_weight_bytes / memory_divisor / mem_bw_bytes_per_cycle)
    mem_w2_cycles = int(w2_weight_bytes / memory_divisor / mem_bw_bytes_per_cycle)
    mem_startup = int(config.memory.per_core_latency_ns * max(config.compute.pe_freq_ghz, 0.1))
    # Keep memory startup in first memory stage so aggregate stays consistent with existing model.
    mem_w1_cycles = max(1, mem_w1_cycles + mem_startup)
    mem_w3_cycles = max(1, mem_w3_cycles)
    mem_w2_cycles = max(1, mem_w2_cycles)

    compute_w1_cycles = _stage_max_gemm_cycles(
        workload=workload,
        op_type="expert_w1_proj",
        config=config,
        strategy=config.workload.partition_strategy,
        shards=shards,
    )
    compute_w3_cycles = _stage_max_gemm_cycles(
        workload=workload,
        op_type="expert_w3_proj",
        config=config,
        strategy=config.workload.partition_strategy,
        shards=shards,
    )
    compute_elem_cycles = _stage_max_elem_cycles(workload=workload, shards=shards)
    compute_w2_cycles = _stage_max_gemm_cycles(
        workload=workload,
        op_type="expert_w2_proj",
        config=config,
        strategy=config.workload.partition_strategy,
        shards=shards,
    )

    network_metrics = _estimate_network_metrics(workload, _fake_mapping(workload, config), config)
    allreduce_cycles = _estimate_allreduce_cycles(workload, config, shards)
    io_injection_cycles = int(network_metrics["io_injection_cycles"])
    network_cycles = int(network_metrics["network_cycles"])

    stages = _build_stages(
        [
            ("io_inject", "io", io_injection_cycles),
            ("mem_w1", "memory", mem_w1_cycles),
            ("compute_w1", "compute", compute_w1_cycles),
            ("mem_w3", "memory", mem_w3_cycles),
            ("compute_w3", "compute", compute_w3_cycles),
            ("compute_silu_elemul", "compute", compute_elem_cycles),
            ("mem_w2", "memory", mem_w2_cycles),
            ("compute_w2", "compute", compute_w2_cycles),
            ("allreduce", "allreduce", allreduce_cycles),
            ("network_combine", "network", network_cycles),
            ("io_eject", "io", io_injection_cycles),
        ]
    )
    total_cycles = stages[-1].end_cycle if stages else 0
    moved_mem_bytes = (w1_weight_bytes + w3_weight_bytes + w2_weight_bytes) / memory_divisor
    peak_mem_bw_utilization = min(1.0, moved_mem_bytes / max(1.0, total_cycles * mem_bw_bytes_per_cycle))

    if not config_label:
        config_label = (
            f"{config.workload.partition_strategy}/s{shards}/b{decode_tokens}/"
            f"{config.network.noc.topology}:{config.network.now.topology}"
        )

    return PipelineBreakdown(
        config_label=config_label,
        stages=stages,
        total_cycles=total_cycles,
        partition_strategy=config.workload.partition_strategy,
        partition_shards=shards,
        batch_size=decode_tokens,
        peak_mem_bw_utilization=peak_mem_bw_utilization,
    )


def _build_stages(stage_specs: list[tuple[str, str, int]]) -> list[PipelineStage]:
    stages: list[PipelineStage] = []
    cursor = 0
    for name, category, duration in stage_specs:
        dur = max(0, int(duration))
        stages.append(PipelineStage(name=name, category=category, start_cycle=cursor, duration_cycles=dur))
        cursor += dur
    return stages


def _memory_bw_bytes_per_cycle(config: WSEConfig) -> float:
    cycles_per_ns = max(config.compute.pe_freq_ghz, 0.1)
    return max(1.0, config.memory.per_core_bandwidth_gbps / cycles_per_ns)


def _cube_gemm_cycles(m: int, n: int, k: int, config: WSEConfig) -> int:
    if config.compute.pe_type != "cube":
        throughput = max(1, config.compute.pe_width * config.compute.pe_width)
        return max(1, (m * n * k) // throughput)

    m_tile = max(1, config.compute.cube_m_tile)
    n_tile = max(1, config.compute.cube_n_tile)
    k_tile = max(1, config.compute.cube_k_tile)
    startup = max(0, config.compute.cube_startup_cycles)
    steady = max(1, config.compute.cube_steady_cycles)

    per_shard_tiles = ceil(m / m_tile) * ceil(n / n_tile) * ceil(k / k_tile)
    if per_shard_tiles <= 0:
        return 0
    per_shard_cycles = startup + max(0, per_shard_tiles - 1) * steady
    return per_shard_cycles


def _stage_max_gemm_cycles(
    workload,
    op_type: str,
    config: WSEConfig,
    strategy: str,
    shards: int,
) -> int:
    stage_max = 0
    for op in workload.ops:
        if op.op_type != op_type:
            continue
        n_eff = op.n
        k_eff = op.k
        if strategy == "col":
            n_eff = ceil(op.n / max(1, shards))
        elif strategy == "k_split":
            k_eff = ceil(op.k / max(1, shards))
        op_cycles = _cube_gemm_cycles(m=op.m, n=n_eff, k=k_eff, config=config)
        stage_max = max(stage_max, op_cycles)
    return stage_max


def _stage_max_elem_cycles(workload, shards: int) -> int:
    stage_max = 0
    for op in workload.ops:
        if op.op_type != "elementwise_mul":
            continue
        op_cycles = (op.m * op.n) // max(1, shards)
        stage_max = max(stage_max, op_cycles)
    return stage_max


class _MappingProxy:
    def __init__(self, assignments: dict[str, list[int]]) -> None:
        self.assignments = assignments
        self.core_tasks: dict[int, list[str]] = {}


def _fake_mapping(workload, config: WSEConfig) -> _MappingProxy:
    """
    Provide a compact mapping for network estimation.

    The network estimator only requires op->core assignments for routed/shared W1 ops.
    """
    total_cores = max(1, config.wafer.total_cores)
    assignments: dict[str, list[int]] = {}
    cursor = 0
    for op in workload.ops:
        if op.op_type != "expert_w1_proj":
            continue
        # Spread each expert over a few cores to avoid degenerate per-core traffic.
        span = min(4, total_cores)
        cores = [(cursor + i) % total_cores for i in range(span)]
        assignments[op.name] = cores
        cursor = (cursor + span) % total_cores
    return _MappingProxy(assignments)

"""Pipeline-oriented analysis for DeepSeek-V4-Pro FFN DSE results."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from wsesim.core.config import WSEConfig
from wsesim.dse.evaluator_deepseek import (
    _effective_partition_shards,
    _estimate_compute_stage_cycles,
    _estimate_memory_stage_cycles,
    _estimate_network_metrics,
    _estimate_stage_overlap_latency,
    _resolve_partitioner,
    _simulate_allreduce_cycles,
)
from wsesim.workload.generator import (
    DeepSeekV4ProFFNProfile,
    generate_deepseek_v4_pro_decode_ffn_workload,
)
from wsesim.workload.partition.base import TileTask


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

    decode_tokens = int(workload.metadata["decode_tokens"])
    mem_bw_bytes_per_cycle = _memory_bw_bytes_per_cycle(config)
    partitioner = _resolve_partitioner(config.workload.partition_strategy)
    tasks: dict[str, list[TileTask]] = {
        op.name: partitioner.partition(
            op,
            shards=(
                shards
                if op.op_type in {"expert_w1_proj", "expert_w3_proj", "expert_w2_proj", "elementwise_mul"}
                else 1
            ),
        )
        for op in workload.ops
    }
    op_lookup = {op.name: op for op in workload.ops}
    compute_stage_cycles = _estimate_compute_stage_cycles(tasks, op_lookup, config)
    memory_stage_cycles = _estimate_memory_stage_cycles(
        workload=workload,
        config=config,
        partition_shards=shards,
        partition_strategy=config.workload.partition_strategy,
    )
    compute_w1_cycles = int(compute_stage_cycles["expert_w1_proj"])
    compute_w3_cycles = int(compute_stage_cycles["expert_w3_proj"])
    compute_elem_cycles = int(compute_stage_cycles["elementwise_mul"])
    compute_w2_cycles = int(compute_stage_cycles["expert_w2_proj"])
    mem_w1_cycles = int(memory_stage_cycles["expert_w1_proj"])
    mem_w3_cycles = int(memory_stage_cycles["expert_w3_proj"])
    mem_w2_cycles = int(memory_stage_cycles["expert_w2_proj"])

    network_metrics = _estimate_network_metrics(workload, _fake_mapping(workload, config), config)
    allreduce_cycles = _simulate_allreduce_cycles(workload, config, shards, _fake_mapping(workload, config))
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
    moved_mem_cycles = mem_w1_cycles + mem_w3_cycles + mem_w2_cycles
    peak_mem_bw_utilization = min(1.0, moved_mem_cycles / max(1.0, total_cycles))

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


class _MappingProxy:
    def __init__(self, assignments: dict[str, list[int]]) -> None:
        self.assignments = assignments
        self.core_tasks: dict[int, list[str]] = {}


def compute_overlap_breakdown(config: WSEConfig) -> dict:
    """Return max-path/stage-overlap latency breakdown aligned with the evaluator."""
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
    partitioner = _resolve_partitioner(config.workload.partition_strategy)
    tasks = {
        op.name: partitioner.partition(
            op,
            shards=(
                shards
                if op.op_type in {"expert_w1_proj", "expert_w3_proj", "expert_w2_proj", "elementwise_mul"}
                else 1
            ),
        )
        for op in workload.ops
    }
    op_lookup = {op.name: op for op in workload.ops}
    compute_stage = _estimate_compute_stage_cycles(tasks, op_lookup, config)
    memory_stage = _estimate_memory_stage_cycles(
        workload=workload,
        config=config,
        partition_shards=shards,
        partition_strategy=config.workload.partition_strategy,
    )
    mapping = _fake_mapping(workload, config)
    network_metrics = _estimate_network_metrics(workload, mapping, config)
    allreduce_cycles = _simulate_allreduce_cycles(workload, config, shards, mapping)

    w1 = max(int(compute_stage["expert_w1_proj"]), int(memory_stage["expert_w1_proj"]))
    w3 = max(int(compute_stage["expert_w3_proj"]), int(memory_stage["expert_w3_proj"]))
    w2 = max(int(compute_stage["expert_w2_proj"]), int(memory_stage["expert_w2_proj"]))
    elem = int(compute_stage["elementwise_mul"])
    ffn_path = max(w1, w3) + elem + w2
    io_cycles = int(network_metrics["io_injection_cycles"])
    net_cycles = int(network_metrics["network_cycles"])
    comm_tail = max(net_cycles, io_cycles, int(allreduce_cycles))
    total = _estimate_stage_overlap_latency(
        compute_stage_cycles=compute_stage,
        memory_stage_cycles=memory_stage,
        network_cycles=net_cycles,
        io_injection_cycles=io_cycles,
        allreduce_cycles=int(allreduce_cycles),
        partition_strategy=config.workload.partition_strategy,
        tile_pipeline=getattr(config.workload, "tile_pipeline", False),
    )

    return {
        "total_latency_cycles": total,
        "ffn_path_cycles": ffn_path,
        "comm_tail_cycles": comm_tail,
        "w1_stage_cycles": w1,
        "w3_stage_cycles": w3,
        "elem_cycles": elem,
        "w2_stage_cycles": w2,
        "compute_w1": int(compute_stage["expert_w1_proj"]),
        "compute_w3": int(compute_stage["expert_w3_proj"]),
        "compute_w2": int(compute_stage["expert_w2_proj"]),
        "compute_elem": elem,
        "mem_w1": int(memory_stage["expert_w1_proj"]),
        "mem_w3": int(memory_stage["expert_w3_proj"]),
        "mem_w2": int(memory_stage["expert_w2_proj"]),
        "io_injection_cycles": io_cycles,
        "network_cycles": net_cycles,
        "allreduce_cycles": int(allreduce_cycles),
        "partition_strategy": config.workload.partition_strategy,
        "partition_shards": shards,
        "batch_size": int(workload.metadata["decode_tokens"]),
        "noc_topology": config.network.noc.topology,
        "now_topology": config.network.now.topology,
        "noc_routing": config.network.noc.routing,
        "now_routing": config.network.now.routing,
        "tile_pipeline": getattr(config.workload, "tile_pipeline", False),
    }


def format_overlap_breakdown_markdown(breakdown: dict, title: str = "Latency Breakdown") -> str:
    """Format overlap breakdown as markdown for reporting."""
    lines = [
        f"# {title}",
        "",
        "## Configuration",
        f"- Partition: `{breakdown['partition_strategy']}` / shards={breakdown['partition_shards']}",
        f"- Batch size: {breakdown['batch_size']}",
        f"- NoC: `{breakdown['noc_topology']}:{breakdown['noc_routing']}`",
        f"- NoW: `{breakdown['now_topology']}:{breakdown['now_routing']}`",
        f"- Tile pipeline: {breakdown['tile_pipeline']}",
        "",
        "## Total Latency (max-path / stage-overlap model)",
        f"- **Total**: {breakdown['total_latency_cycles']:,} cycles",
        f"- FFN path: {breakdown['ffn_path_cycles']:,} cycles",
        f"- Comm tail: {breakdown['comm_tail_cycles']:,} cycles",
        "",
        "## FFN Path Detail",
        "```",
        f"W1 = max(compute, mem) = max({breakdown['compute_w1']:,}, {breakdown['mem_w1']:,}) = {breakdown['w1_stage_cycles']:,}",
        f"W3 = max(compute, mem) = max({breakdown['compute_w3']:,}, {breakdown['mem_w3']:,}) = {breakdown['w3_stage_cycles']:,}",
        f"ElemMul = {breakdown['elem_cycles']:,}",
        f"W2 = max(compute, mem) = max({breakdown['compute_w2']:,}, {breakdown['mem_w2']:,}) = {breakdown['w2_stage_cycles']:,}",
        f"ffn_path = max(W1,W3) + Elem + W2 = max({breakdown['w1_stage_cycles']:,},{breakdown['w3_stage_cycles']:,}) + {breakdown['elem_cycles']:,} + {breakdown['w2_stage_cycles']:,} = {breakdown['ffn_path_cycles']:,}",
        "```",
        "",
        "## Communication Tail",
        "```",
        f"comm_tail = max(network, io, allreduce)",
        f"         = max({breakdown['network_cycles']:,}, {breakdown['io_injection_cycles']:,}, {breakdown['allreduce_cycles']:,})",
        f"         = {breakdown['comm_tail_cycles']:,}",
        "```",
        "",
        "## Verification",
        f"total = ffn_path + comm_tail = {breakdown['ffn_path_cycles']:,} + {breakdown['comm_tail_cycles']:,} = {breakdown['ffn_path_cycles'] + breakdown['comm_tail_cycles']:,}",
    ]
    return "\n".join(lines)


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

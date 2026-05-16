"""DeepSeek-V3 FFN-aware DSE evaluator."""

from __future__ import annotations

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult
from wsesim.workload.generator import (
    DeepSeekV3FFNProfile,
    generate_deepseek_v3_decode_ffn_workload,
)
from wsesim.workload.mapper import ExpertAffinityMapping, ExpertLocalityMapping, NearestNeighborMapping
from wsesim.workload.partition.expert import ExpertPartition


def evaluate_deepseek_v3_ffn(config: WSEConfig) -> SimResult:
    """Evaluate one WSE config using DeepSeek-V3-like FFN decode workload."""
    profile = DeepSeekV3FFNProfile(
        hidden_dim=config.workload.hidden_dim,
        expert_ffn_dim=config.workload.expert_ffn_dim,
        num_routed_experts=config.workload.num_routed_experts,
        num_shared_experts=config.workload.num_shared_experts,
        top_k=config.workload.top_k,
        decode_tokens=config.workload.decode_tokens,
        routing_skew_alpha=config.workload.routing_skew_alpha,
        capacity_factor=config.workload.capacity_factor,
    )
    workload = generate_deepseek_v3_decode_ffn_workload(profile)

    partitioner = ExpertPartition()
    tasks = {op.name: partitioner.partition(op, shards=1) for op in workload.ops}

    alive_cores = list(range(max(1, config.wafer.total_cores)))
    mapper = _resolve_mapper(config.workload.mapping_strategy)
    mapping = mapper.map(workload, tasks, alive_cores)

    compute_cycles = _estimate_compute_cycles(workload, config)
    network_metrics = _estimate_network_metrics(workload, mapping, config)
    memory_stall_cycles = _estimate_memory_stall_cycles(workload, config)

    result = SimResult(
        total_latency_cycles=compute_cycles + network_metrics["network_cycles"] + memory_stall_cycles,
        compute_cycles=compute_cycles,
        network_cycles=network_metrics["network_cycles"],
        memory_stall_cycles=memory_stall_cycles,
        network_avg_latency=network_metrics["network_avg_latency"],
        network_max_latency=network_metrics["network_max_latency"],
        network_throughput=network_metrics["network_throughput"],
        network_saturation=network_metrics["network_saturation"],
        vc_wait_cycles=network_metrics["vc_wait_cycles"],
        buffer_wait_cycles=network_metrics["buffer_wait_cycles"],
        link_wait_cycles=network_metrics["link_wait_cycles"],
        pipeline_cycles=network_metrics["pipeline_cycles"],
        metadata={
            "workload_model": workload.model_name,
            "active_routed_experts": int(workload.metadata["active_routed_experts"]),
            "mapped_cores": len(mapping.core_tasks),
            "mapping_strategy": config.workload.mapping_strategy,
        },
    )
    return result


def _resolve_mapper(name: str):
    if name == "expert_affinity":
        return ExpertAffinityMapping()
    if name == "expert_locality":
        return ExpertLocalityMapping()
    return NearestNeighborMapping()


def _estimate_compute_cycles(workload, config: WSEConfig) -> int:
    pe_width = max(1, config.compute.pe_width)
    if config.compute.pe_type == "vector":
        throughput_per_cycle = pe_width
    else:
        throughput_per_cycle = pe_width * pe_width

    total_mac = 0
    for op in workload.ops:
        if op.op_type in {"expert_up_proj", "expert_down_proj", "router"}:
            total_mac += op.m * op.n * op.k
    return max(1, total_mac // throughput_per_cycle)


def _estimate_network_metrics(workload, mapping, config: WSEConfig) -> dict[str, float | int]:
    fp_bytes = 2  # BF16/FP16
    hidden = int(workload.metadata["hidden_dim"])
    decode_tokens = int(workload.metadata["decode_tokens"])
    top_k = int(workload.metadata["top_k"])
    active_routed = int(workload.metadata["active_routed_experts"])

    coordinator = mapping.assignments.get("deepseek_v3_token_dispatch", [0])[0]
    data_per_token = hidden * fp_bytes
    total_dispatch_bytes = decode_tokens * top_k * data_per_token
    total_combine_bytes = decode_tokens * data_per_token
    total_bytes = total_dispatch_bytes + total_combine_bytes

    # Estimate average "distance" from coordinator to active expert cores.
    routed_cores = [
        cores[0]
        for op_name, cores in mapping.assignments.items()
        if "_expert_routed_" in op_name and "_up_proj" in op_name and cores
    ]
    if routed_cores:
        avg_hops = sum(abs(core - coordinator) for core in routed_cores) / len(routed_cores)
    else:
        avg_hops = 1.0

    flit_bytes = 32
    noc_bw_bytes_per_cycle = max(1, config.network.noc.link_bw_flits_per_cycle * flit_bytes)
    transfer_cycles = int((total_bytes / noc_bw_bytes_per_cycle) * max(1.0, avg_hops))
    pipeline_cycles = int(
        active_routed
        * (
            config.network.noc.rc_latency_cycles
            + config.network.noc.va_latency_cycles
            + config.network.noc.sa_latency_cycles
            + config.network.noc.st_latency_cycles
        )
    )

    contention = active_routed / max(1, len(mapping.core_tasks))
    vc_wait_cycles = int(200 * contention / max(1, config.network.noc.num_vcs))
    buffer_wait_cycles = int(160 * contention * (8 / max(1, config.network.noc.buffer_depth)))
    link_wait_cycles = int(transfer_cycles * (config.network.noc.link_latency_cycles / 8))

    network_cycles = transfer_cycles + pipeline_cycles + vc_wait_cycles + buffer_wait_cycles + link_wait_cycles
    network_throughput = total_bytes / max(1, network_cycles)
    network_saturation = min(1.0, (vc_wait_cycles + buffer_wait_cycles) / max(1, network_cycles))
    avg_pkt_latency = network_cycles / max(1, decode_tokens)

    return {
        "network_cycles": int(network_cycles),
        "network_avg_latency": float(avg_pkt_latency),
        "network_max_latency": float(avg_pkt_latency * 1.5),
        "network_throughput": float(network_throughput),
        "network_saturation": float(network_saturation),
        "vc_wait_cycles": int(vc_wait_cycles),
        "buffer_wait_cycles": int(buffer_wait_cycles),
        "link_wait_cycles": int(link_wait_cycles),
        "pipeline_cycles": int(pipeline_cycles),
    }


def _estimate_memory_stall_cycles(workload, config: WSEConfig) -> int:
    hidden = int(workload.metadata["hidden_dim"])
    decode_tokens = int(workload.metadata["decode_tokens"])
    ffn_dim = int(workload.metadata["expert_ffn_dim"])
    active_routed = int(workload.metadata["active_routed_experts"])
    fp_bytes = 2

    # Approximate read+write footprint for active expert FFN passes.
    bytes_moved = active_routed * decode_tokens * (hidden + ffn_dim + hidden) * fp_bytes

    mem_bw_bytes_per_cycle = max(1.0, (config.memory.peak_bandwidth_gbps * 1e9) / (1e9))
    transfer_cycles = int(bytes_moved / mem_bw_bytes_per_cycle)
    startup_penalty = int(config.memory.base_latency_ns // 10)
    return max(1, transfer_cycles + startup_penalty)

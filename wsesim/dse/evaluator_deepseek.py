"""DeepSeek-V3 FFN-aware DSE evaluator."""

from __future__ import annotations

from collections import defaultdict
from math import isqrt

import simpy

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.flow_control.wormhole import WormholeFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.routing.table_based import TableBasedRouting
from wsesim.network.routing.ugal import UGALRouting
from wsesim.network.topology.flat_butterfly import FlatButterfly
from wsesim.network.topology.mesh2d import Mesh2D
from wsesim.network.topology.torus2d import Torus2D
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
    hidden = int(workload.metadata["hidden_dim"])
    decode_tokens = int(workload.metadata["decode_tokens"])
    top_k = int(workload.metadata["top_k"])
    fp_bytes = 2

    coordinator = mapping.assignments.get("deepseek_v3_token_dispatch", [0])[0]
    dispatch_by_core: dict[int, int] = defaultdict(int)
    combine_by_core: dict[int, int] = defaultdict(int)

    op_lookup = {op.name: op for op in workload.ops}
    for op_name, cores in mapping.assignments.items():
        if not cores:
            continue
        op = op_lookup.get(op_name)
        if op is None or op.expert_kind != "routed" or op.op_type != "expert_up_proj":
            continue
        core = cores[0]
        bytes_for_expert = op.m * hidden * fp_bytes
        dispatch_by_core[core] += bytes_for_expert
        combine_by_core[core] += bytes_for_expert

    # Shared expert traffic: coordinator to/from shared expert cores.
    for op_name, cores in mapping.assignments.items():
        if not cores:
            continue
        op = op_lookup.get(op_name)
        if op is None or op.expert_kind != "shared" or op.op_type != "expert_up_proj":
            continue
        core = cores[0]
        bytes_for_shared = decode_tokens * hidden * fp_bytes
        dispatch_by_core[core] += bytes_for_shared
        combine_by_core[core] += bytes_for_shared

    total_bytes = sum(dispatch_by_core.values()) + sum(combine_by_core.values())
    if total_bytes <= 0:
        return {
            "network_cycles": 0,
            "network_avg_latency": 0.0,
            "network_max_latency": 0.0,
            "network_throughput": 0.0,
            "network_saturation": 0.0,
            "vc_wait_cycles": 0,
            "buffer_wait_cycles": 0,
            "link_wait_cycles": 0,
            "pipeline_cycles": 0,
        }

    # Use a scaled traffic window for tractable inner-loop simulation.
    scale = min(1.0, 8.0 / max(1, decode_tokens * top_k))
    env = simpy.Environment()
    network = _build_noc_network(env, config)

    packet_processes = []
    for core, bytes_count in dispatch_by_core.items():
        if core == coordinator:
            continue
        scaled_bytes = max(32, int(bytes_count * scale))
        pkt = Packet(
            src=coordinator,
            dst=core,
            size_bytes=scaled_bytes,
            payload_type="dispatch",
            creation_time=float(env.now),
        )
        packet_processes.append(env.process(network.send_packet(pkt)))
    for core, bytes_count in combine_by_core.items():
        if core == coordinator:
            continue
        scaled_bytes = max(32, int(bytes_count * scale))
        pkt = Packet(
            src=core,
            dst=coordinator,
            size_bytes=scaled_bytes,
            payload_type="combine",
            creation_time=float(env.now),
        )
        packet_processes.append(env.process(network.send_packet(pkt)))

    if packet_processes:
        env.run()

    simulated_cycles = max(1.0, float(env.now))
    bw_factor = max(1, config.network.noc.link_bw_flits_per_cycle)
    # Extrapolate from traffic window back to full decode traffic.
    network_cycles = int((simulated_cycles / max(scale, 1e-6)) / bw_factor)
    vc_wait_cycles = int((network.stats.vc_wait_cycles / max(scale, 1e-6)) / bw_factor)
    buffer_wait_cycles = int((network.stats.buffer_wait_cycles / max(scale, 1e-6)) / bw_factor)
    link_wait_cycles = int((network.stats.link_wait_cycles / max(scale, 1e-6)) / bw_factor)
    pipeline_cycles = int(network.stats.pipeline_cycles / max(scale, 1e-6))

    network_throughput = total_bytes / max(1.0, float(network_cycles))
    network_saturation = min(
        1.0, (vc_wait_cycles + buffer_wait_cycles) / max(1.0, float(network_cycles))
    )
    packet_count = max(1, network.stats.packets_sent)
    avg_pkt_latency = network_cycles / packet_count

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


def _build_noc_network(env: simpy.Environment, config: WSEConfig) -> UnifiedNetwork:
    num_nodes = max(4, config.wafer.total_cores)
    topology_name = config.network.noc.topology
    if topology_name in {"mesh2d", "torus2d"}:
        side = isqrt(num_nodes)
        if side * side != num_nodes:
            side += 1
            num_nodes = side * side
        topology = Mesh2D() if topology_name == "mesh2d" else Torus2D()
    elif topology_name == "flat_butterfly":
        topology = FlatButterfly()
    else:
        topology = Mesh2D()

    routing_name = config.network.noc.routing
    if routing_name in {"xy", "dimension_order"}:
        routing = DimensionOrderRouting()
    elif routing_name == "ugal":
        routing = UGALRouting(seed=config.dse.random_seed)
    else:
        routing = DimensionOrderRouting()

    flow_name = config.network.noc.flow_control
    if flow_name == "wormhole":
        flow_control = WormholeFlowControl()
    else:
        flow_control = CreditBasedVCFlowControl()

    network = UnifiedNetwork(
        env=env,
        topology=topology,
        routing=routing,
        flow_control=flow_control,
        num_nodes=num_nodes,
        link_bw_flits_per_cycle=config.network.noc.link_bw_flits_per_cycle,
        link_latency_cycles=config.network.noc.link_latency_cycles,
        num_vcs=config.network.noc.num_vcs,
        buffer_depth=config.network.noc.buffer_depth,
        router_pipeline_mode=config.network.noc.router_pipeline_mode,
        rc_latency_cycles=config.network.noc.rc_latency_cycles,
        va_latency_cycles=config.network.noc.va_latency_cycles,
        sa_latency_cycles=config.network.noc.sa_latency_cycles,
        st_latency_cycles=config.network.noc.st_latency_cycles,
        crossbar_bw_flits_per_cycle=config.network.noc.crossbar_bw_flits_per_cycle,
    )
    if routing_name == "table_based":
        network.routing = TableBasedRouting(network.graph)
    return network

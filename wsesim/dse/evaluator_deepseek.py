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
    traffic: list[tuple[int, int, int, str]] = []
    for core, bytes_count in dispatch_by_core.items():
        if core == coordinator:
            continue
        scaled_bytes = max(32, int(bytes_count * scale))
        traffic.append((coordinator, core, scaled_bytes, "dispatch"))
    for core, bytes_count in combine_by_core.items():
        if core == coordinator:
            continue
        scaled_bytes = max(32, int(bytes_count * scale))
        traffic.append((core, coordinator, scaled_bytes, "combine"))

    if not traffic:
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

    simulated_cycles, simulated_stats = _run_hierarchical_network_simulation(traffic, config)
    bw_factor = max(1, config.network.noc.link_bw_flits_per_cycle)
    # Extrapolate from traffic window back to full decode traffic.
    network_cycles = int((simulated_cycles / max(scale, 1e-6)) / bw_factor)
    vc_wait_cycles = int((simulated_stats["vc_wait_cycles"] / max(scale, 1e-6)) / bw_factor)
    buffer_wait_cycles = int((simulated_stats["buffer_wait_cycles"] / max(scale, 1e-6)) / bw_factor)
    link_wait_cycles = int((simulated_stats["link_wait_cycles"] / max(scale, 1e-6)) / bw_factor)
    pipeline_cycles = int(simulated_stats["pipeline_cycles"] / max(scale, 1e-6))

    network_throughput = total_bytes / max(1.0, float(network_cycles))
    network_saturation = min(
        1.0, (vc_wait_cycles + buffer_wait_cycles) / max(1.0, float(network_cycles))
    )
    packet_count = max(1, int(simulated_stats["packets_sent"]))
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
    return _build_network_domain(
        env=env,
        num_nodes=max(4, config.wafer.total_cores),
        domain_config=config.network.noc,
        random_seed=config.dse.random_seed,
    )


def _build_now_network(env: simpy.Environment, config: WSEConfig) -> UnifiedNetwork:
    return _build_network_domain(
        env=env,
        num_nodes=max(4, config.wafer.reticle_count),
        domain_config=config.network.now,
        random_seed=config.dse.random_seed + 17,
    )


def _build_network_domain(env: simpy.Environment, num_nodes: int, domain_config, random_seed: int) -> UnifiedNetwork:
    num_nodes = max(4, num_nodes)
    topology_name = domain_config.topology
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

    routing_name = domain_config.routing
    if routing_name in {"xy", "dimension_order"}:
        routing = DimensionOrderRouting()
    elif routing_name == "ugal":
        routing = UGALRouting(seed=random_seed)
    else:
        routing = DimensionOrderRouting()

    flow_name = domain_config.flow_control
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
        link_bw_flits_per_cycle=domain_config.link_bw_flits_per_cycle,
        link_latency_cycles=domain_config.link_latency_cycles,
        num_vcs=domain_config.num_vcs,
        buffer_depth=domain_config.buffer_depth,
        router_pipeline_mode=domain_config.router_pipeline_mode,
        rc_latency_cycles=domain_config.rc_latency_cycles,
        va_latency_cycles=domain_config.va_latency_cycles,
        sa_latency_cycles=domain_config.sa_latency_cycles,
        st_latency_cycles=domain_config.st_latency_cycles,
        crossbar_bw_flits_per_cycle=domain_config.crossbar_bw_flits_per_cycle,
    )
    if routing_name == "table_based":
        network.routing = TableBasedRouting(network.graph)
    return network


def _run_hierarchical_network_simulation(
    traffic: list[tuple[int, int, int, str]], config: WSEConfig
) -> tuple[float, dict[str, int]]:
    env = simpy.Environment()
    cores_per_reticle = max(1, config.wafer.cores_per_reticle)
    reticle_count = max(1, config.wafer.reticle_count)
    gateways = _reticle_gateways(
        cores_per_reticle=cores_per_reticle,
        gateways_per_reticle=max(1, config.network.gateways_per_reticle),
    )
    noc_networks = {
        reticle: _build_network_domain(
            env=env,
            num_nodes=cores_per_reticle,
            domain_config=config.network.noc,
            random_seed=config.dse.random_seed + reticle,
        )
        for reticle in range(reticle_count)
    }
    now_network = _build_now_network(env, config)

    for src_core, dst_core, size_bytes, payload in traffic:
        env.process(
            _send_hierarchical_packet(
                env=env,
                src_core=src_core,
                dst_core=dst_core,
                size_bytes=size_bytes,
                payload_type=payload,
                cores_per_reticle=cores_per_reticle,
                reticles_x=max(1, config.wafer.reticles_x),
                gateway_policy=config.network.gateway_policy,
                gateways=gateways,
                noc_networks=noc_networks,
                now_network=now_network,
            )
        )
    env.run()

    stats = {
        "packets_sent": now_network.stats.packets_sent,
        "vc_wait_cycles": now_network.stats.vc_wait_cycles,
        "buffer_wait_cycles": now_network.stats.buffer_wait_cycles,
        "link_wait_cycles": now_network.stats.link_wait_cycles,
        "pipeline_cycles": now_network.stats.pipeline_cycles,
    }
    for noc in noc_networks.values():
        stats["packets_sent"] += noc.stats.packets_sent
        stats["vc_wait_cycles"] += noc.stats.vc_wait_cycles
        stats["buffer_wait_cycles"] += noc.stats.buffer_wait_cycles
        stats["link_wait_cycles"] += noc.stats.link_wait_cycles
        stats["pipeline_cycles"] += noc.stats.pipeline_cycles
    return float(env.now), stats


def _send_hierarchical_packet(
    env: simpy.Environment,
    src_core: int,
    dst_core: int,
    size_bytes: int,
    payload_type: str,
    cores_per_reticle: int,
    reticles_x: int,
    gateway_policy: str,
    gateways: list[int],
    noc_networks: dict[int, UnifiedNetwork],
    now_network: UnifiedNetwork,
):
    src_reticle = src_core // cores_per_reticle
    dst_reticle = dst_core // cores_per_reticle
    src_local = src_core % cores_per_reticle
    dst_local = dst_core % cores_per_reticle
    src_gateway_local, dst_gateway_local = _select_gateways(
        src_local=src_local,
        dst_local=dst_local,
        src_reticle=src_reticle,
        dst_reticle=dst_reticle,
        reticles_x=reticles_x,
        gateways=gateways,
        policy=gateway_policy,
    )

    if src_reticle == dst_reticle:
        pkt = Packet(
            src=src_local,
            dst=dst_local,
            size_bytes=size_bytes,
            payload_type=f"{payload_type}_noc_local",
            creation_time=float(env.now),
        )
        yield env.process(noc_networks[src_reticle].send_packet(pkt))
        return

    if src_local != src_gateway_local:
        pkt_egress = Packet(
            src=src_local,
            dst=src_gateway_local,
            size_bytes=size_bytes,
            payload_type=f"{payload_type}_noc_egress",
            creation_time=float(env.now),
        )
        yield env.process(noc_networks[src_reticle].send_packet(pkt_egress))

    pkt_now = Packet(
        src=src_reticle,
        dst=dst_reticle,
        size_bytes=size_bytes,
        payload_type=f"{payload_type}_now",
        creation_time=float(env.now),
    )
    yield env.process(now_network.send_packet(pkt_now))

    if dst_local != dst_gateway_local:
        pkt_ingress = Packet(
            src=dst_gateway_local,
            dst=dst_local,
            size_bytes=size_bytes,
            payload_type=f"{payload_type}_noc_ingress",
            creation_time=float(env.now),
        )
        yield env.process(noc_networks[dst_reticle].send_packet(pkt_ingress))


def _reticle_gateways(cores_per_reticle: int, gateways_per_reticle: int) -> list[int]:
    if gateways_per_reticle <= 1:
        return [0]
    stride = max(1, cores_per_reticle // gateways_per_reticle)
    gateways = sorted({min(cores_per_reticle - 1, idx * stride) for idx in range(gateways_per_reticle)})
    if 0 not in gateways:
        gateways.insert(0, 0)
    return gateways


def _select_gateways(
    src_local: int,
    dst_local: int,
    src_reticle: int,
    dst_reticle: int,
    reticles_x: int,
    gateways: list[int],
    policy: str,
) -> tuple[int, int]:
    if not gateways:
        return 0, 0

    if policy != "nearest":
        # Fallback deterministic policy for unsupported options.
        return gateways[0], gateways[0]

    src_coord = _reticle_coord(src_reticle, reticles_x)
    dst_coord = _reticle_coord(dst_reticle, reticles_x)
    now_distance = abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])

    best_pair = (gateways[0], gateways[0])
    best_cost = float("inf")
    for src_gw in gateways:
        for dst_gw in gateways:
            # NoW distance is reticle-level (gateway choice does not change it), but
            # gateway choice changes intra-reticle NoC ingress/egress costs.
            cost = abs(src_local - src_gw) + now_distance + abs(dst_local - dst_gw)
            if cost < best_cost:
                best_cost = cost
                best_pair = (src_gw, dst_gw)
    return best_pair


def _reticle_coord(reticle_id: int, reticles_x: int) -> tuple[int, int]:
    return reticle_id // reticles_x, reticle_id % reticles_x

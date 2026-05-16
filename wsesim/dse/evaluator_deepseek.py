"""DeepSeek-V4-Pro FFN-aware DSE evaluator."""

from __future__ import annotations

from collections import defaultdict
from math import ceil, isqrt

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
    DeepSeekV4ProFFNProfile,
    generate_deepseek_v4_pro_decode_ffn_workload,
)
from wsesim.workload.mapper import ExpertAffinityMapping, ExpertLocalityMapping, NearestNeighborMapping
from wsesim.workload.partition.base import PartitionStrategy, TileTask
from wsesim.workload.partition.block import BlockPartition
from wsesim.workload.partition.col import ColPartition
from wsesim.workload.partition.expert import ExpertPartition
from wsesim.workload.partition.k_split import KPartition
from wsesim.workload.partition.row import RowPartition


def evaluate_deepseek_v4_pro_ffn(config: WSEConfig) -> SimResult:
    """Evaluate one WSE config using DeepSeek-V4-Pro FFN decode workload."""
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

    partitioner = _resolve_partitioner(config.workload.partition_strategy)
    partition_shards = _effective_partition_shards(workload, config)
    tasks = {
        op.name: partitioner.partition(
            op,
            shards=(
                partition_shards
                if op.op_type in {"expert_w1_proj", "expert_w3_proj", "expert_w2_proj", "elementwise_mul"}
                else 1
            ),
        )
        for op in workload.ops
    }

    alive_cores = list(range(max(1, config.wafer.total_cores)))
    mapper = _resolve_mapper(config.workload.mapping_strategy)
    mapping = mapper.map(workload, tasks, alive_cores)

    op_lookup = {op.name: op for op in workload.ops}
    compute_cycles = _estimate_compute_cycles(tasks, op_lookup, config)
    network_metrics = _estimate_network_metrics(workload, mapping, config)
    memory_stall_cycles = _estimate_memory_stall_cycles(workload, config, partition_shards)
    allreduce_cycles = _estimate_allreduce_cycles(workload, config, partition_shards)

    result = SimResult(
        total_latency_cycles=(
            compute_cycles
            + network_metrics["network_cycles"]
            + network_metrics["io_injection_cycles"]
            + memory_stall_cycles
            + allreduce_cycles
        ),
        compute_cycles=compute_cycles,
        network_cycles=network_metrics["network_cycles"],
        io_injection_cycles=int(network_metrics["io_injection_cycles"]),
        memory_stall_cycles=memory_stall_cycles,
        allreduce_cycles=allreduce_cycles,
        network_avg_latency=network_metrics["network_avg_latency"],
        network_max_latency=network_metrics["network_max_latency"],
        network_throughput=network_metrics["network_throughput"],
        network_saturation=network_metrics["network_saturation"],
        vc_wait_cycles=network_metrics["vc_wait_cycles"],
        buffer_wait_cycles=network_metrics["buffer_wait_cycles"],
        link_wait_cycles=network_metrics["link_wait_cycles"],
        pipeline_cycles=network_metrics["pipeline_cycles"],
        gateway_noc_hops=network_metrics["gateway_noc_hops"],
        gateway_peak_load=network_metrics["gateway_peak_load"],
        metadata={
            "workload_model": workload.model_name,
            "active_routed_experts": int(workload.metadata["active_routed_experts"]),
            "mapped_cores": len(mapping.core_tasks),
            "mapping_strategy": config.workload.mapping_strategy,
            "gateway_noc_hops": int(network_metrics.get("gateway_noc_hops", 0)),
            "gateway_peak_load": int(network_metrics.get("gateway_peak_load", 0)),
            "gateway_policy": config.network.gateway_policy,
            "io_distribution_policy": config.network.io_distribution_policy,
            "io_injection_cycles": int(network_metrics.get("io_injection_cycles", 0)),
            "partition_strategy": config.workload.partition_strategy,
            "partition_shards": partition_shards,
            "allreduce_cycles": allreduce_cycles,
        },
    )
    return result


def _resolve_mapper(name: str):
    if name == "expert_affinity":
        return ExpertAffinityMapping()
    if name == "expert_locality":
        return ExpertLocalityMapping()
    return NearestNeighborMapping()


def _resolve_partitioner(name: str) -> PartitionStrategy:
    if name == "row":
        return RowPartition()
    if name == "col":
        return ColPartition()
    if name == "block":
        return BlockPartition()
    if name == "k_split":
        return KPartition()
    return ExpertPartition()


def _effective_partition_shards(workload, config: WSEConfig) -> int:
    if config.workload.partition_strategy == "expert":
        return 1
    requested = max(1, int(config.workload.partition_shards))
    active_experts = int(workload.metadata.get("active_routed_experts", 0)) + int(
        workload.metadata.get("num_shared_experts", 0)
    )
    max_by_cores = max(1, config.wafer.total_cores // max(1, active_experts))
    return max(1, min(requested, max_by_cores))


def _estimate_compute_cycles(
    tasks_by_op: dict[str, list[TileTask]], op_lookup: dict[str, object], config: WSEConfig
) -> int:
    total_cycles = 0
    if config.compute.pe_type == "cube":
        m_tile = max(1, config.compute.cube_m_tile)
        k_tile = max(1, config.compute.cube_k_tile)
        n_tile = max(1, config.compute.cube_n_tile)
        startup_cycles = max(0, config.compute.cube_startup_cycles)
        steady_cycles = max(1, config.compute.cube_steady_cycles)

        for op_name, op_tasks in tasks_by_op.items():
            op = op_lookup.get(op_name)
            if op is None:
                continue
            if op.op_type in {"expert_w1_proj", "expert_w3_proj", "expert_w2_proj", "router"}:
                for task in op_tasks:
                    num_tiles = ceil(task.m / m_tile) * ceil(task.n / n_tile) * ceil(task.k / k_tile)
                    if num_tiles > 0:
                        total_cycles += startup_cycles + max(0, num_tiles - 1) * steady_cycles
            elif op.op_type == "elementwise_mul":
                for task in op_tasks:
                    total_cycles += task.m * task.n
        return max(1, total_cycles)

    pe_width = max(1, config.compute.pe_width)
    throughput_per_cycle = pe_width if config.compute.pe_type == "vector" else pe_width * pe_width
    total_mac = 0
    for op_name, op_tasks in tasks_by_op.items():
        op = op_lookup.get(op_name)
        if op is None:
            continue
        if op.op_type in {"expert_w1_proj", "expert_w3_proj", "expert_w2_proj", "router"}:
            for task in op_tasks:
                total_mac += task.m * task.n * task.k
        elif op.op_type == "elementwise_mul":
            for task in op_tasks:
                total_mac += task.m * task.n
    return max(1, total_mac // throughput_per_cycle)


def _estimate_network_metrics(workload, mapping, config: WSEConfig) -> dict[str, float | int]:
    hidden = int(workload.metadata["hidden_dim"])
    decode_tokens = int(workload.metadata["decode_tokens"])
    top_k = int(workload.metadata["top_k"])
    fp_bytes = 2

    cores_per_reticle = max(1, config.wafer.cores_per_reticle)
    compute_nodes = _reticle_compute_nodes(config)
    io_nodes = _reticle_io_nodes(config)
    if not io_nodes:
        io_nodes = [0]
    dispatch_by_core: dict[int, int] = defaultdict(int)
    combine_by_core: dict[int, int] = defaultdict(int)

    op_lookup = {op.name: op for op in workload.ops}
    for op_name, cores in mapping.assignments.items():
        if not cores:
            continue
        op = op_lookup.get(op_name)
        if op is None or op.expert_kind != "routed" or op.op_type != "expert_w1_proj":
            continue
        bytes_for_expert = op.m * hidden * fp_bytes
        bytes_per_core = max(1, int(bytes_for_expert / max(1, len(cores))))
        for core in cores:
            dispatch_by_core[core] += bytes_per_core
            combine_by_core[core] += bytes_per_core

    # Shared expert traffic: coordinator to/from shared expert cores.
    for op_name, cores in mapping.assignments.items():
        if not cores:
            continue
        op = op_lookup.get(op_name)
        if op is None or op.expert_kind != "shared" or op.op_type != "expert_w1_proj":
            continue
        bytes_for_shared = decode_tokens * hidden * fp_bytes
        bytes_per_core = max(1, int(bytes_for_shared / max(1, len(cores))))
        for core in cores:
            dispatch_by_core[core] += bytes_per_core
            combine_by_core[core] += bytes_per_core

    total_bytes = sum(dispatch_by_core.values()) + sum(combine_by_core.values())
    if total_bytes <= 0:
        return {
            "network_cycles": 0,
            "io_injection_cycles": 0,
            "network_avg_latency": 0.0,
            "network_max_latency": 0.0,
            "network_throughput": 0.0,
            "network_saturation": 0.0,
            "vc_wait_cycles": 0,
            "buffer_wait_cycles": 0,
            "link_wait_cycles": 0,
            "pipeline_cycles": 0,
            "gateway_noc_hops": 0,
            "gateway_peak_load": 0,
        }

    # Use a scaled traffic window for tractable inner-loop simulation.
    scale = min(1.0, 8.0 / max(1, decode_tokens * top_k))
    traffic: list[dict[str, int | str | None]] = []
    dispatch_io_load: dict[tuple[int, int], int] = defaultdict(int)
    combine_io_load: dict[tuple[int, int], int] = defaultdict(int)
    for core, bytes_count in dispatch_by_core.items():
        core_reticle = core // cores_per_reticle
        core_local = core % cores_per_reticle
        io_phys = _assign_io_node(
            core_local=core_local,
            io_nodes=io_nodes,
            policy=config.network.io_distribution_policy,
            io_load={io: dispatch_io_load[(core_reticle, io)] for io in io_nodes},
            compute_nodes=compute_nodes,
            reticle_cols=max(1, config.wafer.reticle_cols),
        )
        dispatch_io_load[(core_reticle, io_phys)] += bytes_count
        scaled_bytes = max(32, int(bytes_count * scale))
        traffic.append(
            {
                "src_core": None,
                "dst_core": core,
                "src_io_phys": io_phys,
                "dst_io_phys": None,
                "size_bytes": scaled_bytes,
                "payload": "dispatch",
            }
        )
    for core, bytes_count in combine_by_core.items():
        core_reticle = core // cores_per_reticle
        core_local = core % cores_per_reticle
        io_phys = _assign_io_node(
            core_local=core_local,
            io_nodes=io_nodes,
            policy=config.network.io_distribution_policy,
            io_load={io: combine_io_load[(core_reticle, io)] for io in io_nodes},
            compute_nodes=compute_nodes,
            reticle_cols=max(1, config.wafer.reticle_cols),
        )
        combine_io_load[(core_reticle, io_phys)] += bytes_count
        scaled_bytes = max(32, int(bytes_count * scale))
        traffic.append(
            {
                "src_core": core,
                "dst_core": None,
                "src_io_phys": None,
                "dst_io_phys": io_phys,
                "size_bytes": scaled_bytes,
                "payload": "combine",
            }
        )

    io_total_bytes: dict[tuple[int, int], int] = defaultdict(int)
    for key, value in dispatch_io_load.items():
        io_total_bytes[key] += value
    for key, value in combine_io_load.items():
        io_total_bytes[key] += value

    if not traffic:
        return {
            "network_cycles": 0,
            "io_injection_cycles": 0,
            "network_avg_latency": 0.0,
            "network_max_latency": 0.0,
            "network_throughput": 0.0,
            "network_saturation": 0.0,
            "vc_wait_cycles": 0,
            "buffer_wait_cycles": 0,
            "link_wait_cycles": 0,
            "pipeline_cycles": 0,
            "gateway_noc_hops": 0,
            "gateway_peak_load": 0,
        }

    simulated_cycles, simulated_stats = _run_hierarchical_network_simulation(traffic, config)
    bw_factor = max(1, config.network.noc.link_bw_flits_per_cycle)
    # Extrapolate from traffic window back to full decode traffic.
    network_cycles = int((simulated_cycles / max(scale, 1e-6)) / bw_factor)
    vc_wait_cycles = int((simulated_stats["vc_wait_cycles"] / max(scale, 1e-6)) / bw_factor)
    buffer_wait_cycles = int((simulated_stats["buffer_wait_cycles"] / max(scale, 1e-6)) / bw_factor)
    link_wait_cycles = int((simulated_stats["link_wait_cycles"] / max(scale, 1e-6)) / bw_factor)
    pipeline_cycles = int(simulated_stats["pipeline_cycles"] / max(scale, 1e-6))
    io_peak_bytes = max(io_total_bytes.values()) if io_total_bytes else 0
    io_bw_bytes_per_cycle = max(1.0, config.wafer.io_bandwidth_gbps / max(config.network.noc.freq_ghz, 0.1))
    io_injection_cycles = int(io_peak_bytes / io_bw_bytes_per_cycle)

    network_throughput = total_bytes / max(1.0, float(network_cycles))
    network_saturation = min(
        1.0, (vc_wait_cycles + buffer_wait_cycles) / max(1.0, float(network_cycles))
    )
    packet_count = max(1, int(simulated_stats["packets_sent"]))
    avg_pkt_latency = network_cycles / packet_count

    return {
        "network_cycles": int(network_cycles),
        "io_injection_cycles": int(io_injection_cycles),
        "network_avg_latency": float(avg_pkt_latency),
        "network_max_latency": float(avg_pkt_latency * 1.5),
        "network_throughput": float(network_throughput),
        "network_saturation": float(network_saturation),
        "vc_wait_cycles": int(vc_wait_cycles),
        "buffer_wait_cycles": int(buffer_wait_cycles),
        "link_wait_cycles": int(link_wait_cycles),
        "pipeline_cycles": int(pipeline_cycles),
        "gateway_noc_hops": int(simulated_stats["gateway_noc_hops"]),
        "gateway_peak_load": int(simulated_stats["gateway_peak_load"]),
    }


def _estimate_memory_stall_cycles(workload, config: WSEConfig, partition_shards: int = 1) -> int:
    hidden = int(workload.metadata["hidden_dim"])
    decode_tokens = int(workload.metadata["decode_tokens"])
    ffn_dim = int(workload.metadata["expert_ffn_dim"])
    active_routed = int(workload.metadata["active_routed_experts"])
    fp_bytes = 2

    # Approximate read+write footprint for active expert FFN passes.
    # V4-Pro expert path uses W1/W3 (up-projections), elementwise fuse, then W2 down-proj.
    bytes_moved = active_routed * decode_tokens * (hidden + 2 * ffn_dim + hidden) * fp_bytes
    bytes_moved = int(bytes_moved / max(1, partition_shards))

    cycles_per_ns = max(config.compute.pe_freq_ghz, 0.1)
    mem_bw_bytes_per_cycle = max(1.0, config.memory.per_core_bandwidth_gbps / cycles_per_ns)
    transfer_cycles = int(bytes_moved / mem_bw_bytes_per_cycle)
    startup_penalty = int(config.memory.per_core_latency_ns * cycles_per_ns)
    return max(1, transfer_cycles + startup_penalty)


def _estimate_allreduce_cycles(workload, config: WSEConfig, partition_shards: int) -> int:
    shards = max(1, partition_shards)
    strategy = config.workload.partition_strategy
    if shards <= 1 or strategy == "expert":
        return 0

    decode_tokens = int(workload.metadata["decode_tokens"])
    hidden = int(workload.metadata["hidden_dim"])
    ffn_dim = int(workload.metadata["expert_ffn_dim"])
    active_experts = int(workload.metadata["active_routed_experts"]) + int(
        workload.metadata["num_shared_experts"]
    )
    fp_bytes = 2

    if strategy == "col":
        payload_bytes = decode_tokens * hidden * fp_bytes * active_experts
    elif strategy == "k_split":
        payload_bytes = 2 * decode_tokens * ffn_dim * fp_bytes * active_experts
    else:
        return 0

    noc_bytes_per_cycle = max(
        1.0,
        config.network.noc.link_width_bytes * config.network.noc.link_bw_flits_per_cycle,
    )
    ring_factor = 2.0 * (shards - 1) / shards
    transfer_cycles = int(ring_factor * payload_bytes / noc_bytes_per_cycle)
    latency_cycles = int(2 * (shards - 1) * max(1, config.network.noc.link_latency_cycles))
    return max(0, transfer_cycles + latency_cycles)


def _build_noc_network(env: simpy.Environment, config: WSEConfig) -> UnifiedNetwork:
    return _build_network_domain(
        env=env,
        num_nodes=max(4, config.wafer.reticle_rows * config.wafer.reticle_cols),
        domain_config=config.network.noc,
        random_seed=config.dse.random_seed,
        rows=max(1, config.wafer.reticle_rows),
        cols=max(1, config.wafer.reticle_cols),
    )


def _build_now_network(env: simpy.Environment, config: WSEConfig) -> UnifiedNetwork:
    return _build_network_domain(
        env=env,
        num_nodes=max(4, config.wafer.reticle_count),
        domain_config=config.network.now,
        random_seed=config.dse.random_seed + 17,
    )


def _build_network_domain(
    env: simpy.Environment,
    num_nodes: int,
    domain_config,
    random_seed: int,
    rows: int | None = None,
    cols: int | None = None,
) -> UnifiedNetwork:
    num_nodes = max(4, num_nodes)
    topology_name = domain_config.topology
    if topology_name in {"mesh2d", "torus2d"}:
        if rows is not None and cols is not None and topology_name == "mesh2d":
            num_nodes = max(4, rows * cols)
            topology = Mesh2D(rows=rows, cols=cols)
        else:
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
        flit_bytes=domain_config.link_width_bytes,
    )
    if routing_name == "table_based":
        network.routing = TableBasedRouting(network.graph)
    return network


def _run_hierarchical_network_simulation(
    traffic: list[dict[str, int | str | None]], config: WSEConfig
) -> tuple[float, dict[str, int]]:
    env = simpy.Environment()
    cores_per_reticle = max(1, config.wafer.cores_per_reticle)
    reticle_rows = max(1, config.wafer.reticle_rows)
    reticle_cols = max(1, config.wafer.reticle_cols)
    compute_nodes = _reticle_compute_nodes(config)
    reticle_count = max(1, config.wafer.reticle_count)
    gateways = _reticle_gateways(
        cores_per_reticle=cores_per_reticle,
        gateways_per_reticle=max(1, config.network.gateways_per_reticle),
    )
    gateway_load: dict[tuple[int, int], int] = defaultdict(int)
    noc_networks = {
        reticle: _build_network_domain(
            env=env,
            num_nodes=reticle_rows * reticle_cols,
            domain_config=config.network.noc,
            random_seed=config.dse.random_seed + reticle,
            rows=reticle_rows,
            cols=reticle_cols,
        )
        for reticle in range(reticle_count)
    }
    now_network = _build_now_network(env, config)

    for item in traffic:
        env.process(
            _send_hierarchical_packet(
                env=env,
                src_core=item["src_core"] if isinstance(item["src_core"], int) else None,
                dst_core=item["dst_core"] if isinstance(item["dst_core"], int) else None,
                src_io_phys=item["src_io_phys"] if isinstance(item["src_io_phys"], int) else None,
                dst_io_phys=item["dst_io_phys"] if isinstance(item["dst_io_phys"], int) else None,
                size_bytes=int(item["size_bytes"]),
                payload_type=str(item["payload"]),
                cores_per_reticle=cores_per_reticle,
                reticles_x=max(1, config.wafer.reticles_x),
                gateway_policy=config.network.gateway_policy,
                gateways=gateways,
                gateway_load=gateway_load,
                compute_nodes=compute_nodes,
                reticle_cols=reticle_cols,
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
        "gateway_noc_hops": 0,
        "gateway_peak_load": 0,
    }
    for noc in noc_networks.values():
        stats["packets_sent"] += noc.stats.packets_sent
        stats["vc_wait_cycles"] += noc.stats.vc_wait_cycles
        stats["buffer_wait_cycles"] += noc.stats.buffer_wait_cycles
        stats["link_wait_cycles"] += noc.stats.link_wait_cycles
        stats["pipeline_cycles"] += noc.stats.pipeline_cycles
    for item in traffic:
        src_core = item["src_core"] if isinstance(item["src_core"], int) else None
        dst_core = item["dst_core"] if isinstance(item["dst_core"], int) else None
        src_io_phys = item["src_io_phys"] if isinstance(item["src_io_phys"], int) else None
        dst_io_phys = item["dst_io_phys"] if isinstance(item["dst_io_phys"], int) else None

        if src_core is None and dst_core is not None and src_io_phys is not None:
            dst_local = dst_core % cores_per_reticle
            dst_phys = _logical_to_physical(dst_local, compute_nodes)
            stats["gateway_noc_hops"] += _physical_noc_distance(
                src_io_phys, dst_phys, reticle_cols=reticle_cols
            )
            continue

        if src_core is not None and dst_core is None and dst_io_phys is not None:
            src_local = src_core % cores_per_reticle
            src_phys = _logical_to_physical(src_local, compute_nodes)
            stats["gateway_noc_hops"] += _physical_noc_distance(
                src_phys, dst_io_phys, reticle_cols=reticle_cols
            )
            continue

        if src_core is None or dst_core is None:
            continue
        src_reticle = src_core // cores_per_reticle
        dst_reticle = dst_core // cores_per_reticle
        src_local = src_core % cores_per_reticle
        dst_local = dst_core % cores_per_reticle
        if src_reticle == dst_reticle:
            stats["gateway_noc_hops"] += abs(src_local - dst_local)
            continue
        src_gw, dst_gw = _select_gateways(
            src_local=src_local,
            dst_local=dst_local,
            src_reticle=src_reticle,
            dst_reticle=dst_reticle,
            reticles_x=max(1, config.wafer.reticles_x),
            gateways=gateways,
            policy=config.network.gateway_policy,
            gateway_load=gateway_load,
            compute_nodes=compute_nodes,
            reticle_cols=reticle_cols,
        )
        stats["gateway_noc_hops"] += _logical_noc_distance(
            src_local,
            src_gw,
            active_nodes=compute_nodes,
            reticle_cols=reticle_cols,
        )
        stats["gateway_noc_hops"] += _logical_noc_distance(
            dst_local,
            dst_gw,
            active_nodes=compute_nodes,
            reticle_cols=reticle_cols,
        )
    stats["gateway_peak_load"] = max(gateway_load.values()) if gateway_load else 0
    return float(env.now), stats


def _send_hierarchical_packet(
    env: simpy.Environment,
    src_core: int | None,
    dst_core: int | None,
    src_io_phys: int | None,
    dst_io_phys: int | None,
    size_bytes: int,
    payload_type: str,
    cores_per_reticle: int,
    reticles_x: int,
    gateway_policy: str,
    gateways: list[int],
    gateway_load: dict[tuple[int, int], int],
    compute_nodes: list[int],
    reticle_cols: int,
    noc_networks: dict[int, UnifiedNetwork],
    now_network: UnifiedNetwork,
):
    if src_core is None and dst_core is not None and src_io_phys is not None:
        dst_reticle = dst_core // cores_per_reticle
        dst_local = dst_core % cores_per_reticle
        dst_phys = _logical_to_physical(dst_local, compute_nodes)
        pkt = Packet(
            src=src_io_phys,
            dst=dst_phys,
            size_bytes=size_bytes,
            payload_type=f"{payload_type}_noc_inject",
            creation_time=float(env.now),
        )
        yield env.process(noc_networks[dst_reticle].send_packet(pkt))
        return

    if src_core is not None and dst_core is None and dst_io_phys is not None:
        src_reticle = src_core // cores_per_reticle
        src_local = src_core % cores_per_reticle
        src_phys = _logical_to_physical(src_local, compute_nodes)
        pkt = Packet(
            src=src_phys,
            dst=dst_io_phys,
            size_bytes=size_bytes,
            payload_type=f"{payload_type}_noc_eject",
            creation_time=float(env.now),
        )
        yield env.process(noc_networks[src_reticle].send_packet(pkt))
        return

    if src_core is None or dst_core is None:
        return

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
        gateway_load=gateway_load,
        compute_nodes=compute_nodes,
        reticle_cols=reticle_cols,
    )

    if src_reticle == dst_reticle:
        src_phys = _logical_to_physical(src_local, compute_nodes)
        dst_phys = _logical_to_physical(dst_local, compute_nodes)
        pkt = Packet(
            src=src_phys,
            dst=dst_phys,
            size_bytes=size_bytes,
            payload_type=f"{payload_type}_noc_local",
            creation_time=float(env.now),
        )
        yield env.process(noc_networks[src_reticle].send_packet(pkt))
        return

    gateway_load[(src_reticle, src_gateway_local)] += 1
    gateway_load[(dst_reticle, dst_gateway_local)] += 1

    if src_local != src_gateway_local:
        src_phys = _logical_to_physical(src_local, compute_nodes)
        src_gw_phys = _logical_to_physical(src_gateway_local, compute_nodes)
        pkt_egress = Packet(
            src=src_phys,
            dst=src_gw_phys,
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
        dst_phys = _logical_to_physical(dst_local, compute_nodes)
        dst_gw_phys = _logical_to_physical(dst_gateway_local, compute_nodes)
        pkt_ingress = Packet(
            src=dst_gw_phys,
            dst=dst_phys,
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
    gateway_load: dict[tuple[int, int], int] | None = None,
    compute_nodes: list[int] | None = None,
    reticle_cols: int = 1,
) -> tuple[int, int]:
    if not gateways:
        return 0, 0

    src_coord = _reticle_coord(src_reticle, reticles_x)
    dst_coord = _reticle_coord(dst_reticle, reticles_x)
    now_distance = abs(src_coord[0] - dst_coord[0]) + abs(src_coord[1] - dst_coord[1])

    best_pair = (gateways[0], gateways[0])
    best_cost = float("inf")
    gateway_load = gateway_load or {}
    for src_gw in gateways:
        for dst_gw in gateways:
            # NoW distance is reticle-level (gateway choice does not change it), but
            # gateway choice changes intra-reticle NoC ingress/egress costs.
            src_to_gw = _logical_noc_distance(
                src_local, src_gw, active_nodes=compute_nodes, reticle_cols=reticle_cols
            )
            dst_to_gw = _logical_noc_distance(
                dst_local, dst_gw, active_nodes=compute_nodes, reticle_cols=reticle_cols
            )
            cost = src_to_gw + now_distance + dst_to_gw
            if policy == "load_aware":
                cost += 0.5 * (
                    gateway_load.get((src_reticle, src_gw), 0)
                    + gateway_load.get((dst_reticle, dst_gw), 0)
                )
            if cost < best_cost:
                best_cost = cost
                best_pair = (src_gw, dst_gw)
    return best_pair


def _reticle_coord(reticle_id: int, reticles_x: int) -> tuple[int, int]:
    return reticle_id // reticles_x, reticle_id % reticles_x


def _reticle_dead_nodes(config: WSEConfig) -> set[int]:
    cols = max(1, config.wafer.reticle_cols)
    return {r * cols + c for r, c in config.wafer.reticle_dead_positions}


def _reticle_io_nodes(config: WSEConfig) -> list[int]:
    cols = max(1, config.wafer.reticle_cols)
    io_nodes = [r * cols + c for r, c in config.wafer.reticle_io_positions]
    return sorted(set(io_nodes))


def _reticle_active_nodes(config: WSEConfig) -> list[int]:
    rows = max(1, config.wafer.reticle_rows)
    cols = max(1, config.wafer.reticle_cols)
    dead_nodes = _reticle_dead_nodes(config)
    active = [node for node in range(rows * cols) if node not in dead_nodes]
    return active


def _reticle_compute_nodes(config: WSEConfig) -> list[int]:
    io_nodes = set(_reticle_io_nodes(config))
    return [node for node in _reticle_active_nodes(config) if node not in io_nodes]


def _logical_to_physical(local_idx: int, physical_nodes: list[int]) -> int:
    if not physical_nodes:
        return 0
    bounded = min(max(local_idx, 0), len(physical_nodes) - 1)
    return physical_nodes[bounded]


def _logical_noc_distance(
    src_local: int,
    dst_local: int,
    active_nodes: list[int] | None,
    reticle_cols: int,
) -> int:
    if not active_nodes:
        return abs(src_local - dst_local)
    src_phys = _logical_to_physical(src_local, active_nodes)
    dst_phys = _logical_to_physical(dst_local, active_nodes)
    src_row, src_col = divmod(src_phys, max(1, reticle_cols))
    dst_row, dst_col = divmod(dst_phys, max(1, reticle_cols))
    return abs(src_row - dst_row) + abs(src_col - dst_col)


def _physical_noc_distance(src_phys: int, dst_phys: int, reticle_cols: int) -> int:
    src_row, src_col = divmod(src_phys, max(1, reticle_cols))
    dst_row, dst_col = divmod(dst_phys, max(1, reticle_cols))
    return abs(src_row - dst_row) + abs(src_col - dst_col)


def _assign_io_node(
    core_local: int,
    io_nodes: list[int],
    policy: str,
    io_load: dict[int, int],
    compute_nodes: list[int],
    reticle_cols: int,
) -> int:
    if not io_nodes:
        return 0
    if policy == "round_robin":
        return io_nodes[core_local % len(io_nodes)]

    core_phys = _logical_to_physical(core_local, compute_nodes)
    if policy == "nearest":
        return min(
            io_nodes,
            key=lambda io: (_physical_noc_distance(core_phys, io, reticle_cols), io),
        )

    return min(
        io_nodes,
        key=lambda io: (io_load.get(io, 0), _physical_noc_distance(core_phys, io, reticle_cols), io),
    )


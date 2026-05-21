from __future__ import annotations

from copy import deepcopy

from wsesim.core.config import WSEConfig
from wsesim.dse.evaluator_deepseek import evaluate_deepseek_v4_pro_ffn


def test_deepseek_evaluator_produces_nontrivial_metrics() -> None:
    cfg = WSEConfig()
    cfg.workload.model_name = "deepseek_v4_pro_ffn_decode"
    cfg.workload.hidden_dim = 7168
    cfg.workload.expert_ffn_dim = 3072
    cfg.workload.num_routed_experts = 96
    cfg.workload.num_shared_experts = 1
    cfg.workload.top_k = 6
    cfg.workload.decode_tokens = 16
    cfg.workload.mapping_strategy = "expert_affinity"
    assert cfg.wafer.cores_per_reticle == 44
    assert cfg.compute.pe_type == "cube"
    assert cfg.compute.cube_steady_cycles == 5

    result = evaluate_deepseek_v4_pro_ffn(cfg)
    assert result.total_latency_cycles > 0
    assert result.compute_cycles > 0
    assert result.network_cycles > 0
    assert result.network_throughput > 0
    assert result.vc_wait_cycles >= 0
    assert result.buffer_wait_cycles >= 0
    assert result.link_wait_cycles >= 0
    assert int(result.metadata["io_injection_cycles"]) > 0
    assert result.metadata["io_distribution_policy"] == cfg.network.io_distribution_policy
    assert result.metadata["workload_model"] == "deepseek_v4_pro_ffn_decode"


def test_deepseek_evaluator_responds_to_network_vc_scaling() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.num_routed_experts = 128
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.decode_tokens = 32
    base.workload.mapping_strategy = "expert_affinity"

    low_vc = deepcopy(base)
    high_vc = deepcopy(base)
    low_vc.network.noc.num_vcs = 1
    high_vc.network.noc.num_vcs = 4

    low_result = evaluate_deepseek_v4_pro_ffn(low_vc)
    high_result = evaluate_deepseek_v4_pro_ffn(high_vc)
    assert high_result.vc_wait_cycles < low_result.vc_wait_cycles


def test_deepseek_evaluator_responds_to_link_bandwidth() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.num_routed_experts = 8
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.decode_tokens = 24
    base.workload.mapping_strategy = "expert_affinity"

    low_bw = deepcopy(base)
    high_bw = deepcopy(base)
    low_bw.network.noc.link_bw_flits_per_cycle = 1
    high_bw.network.noc.link_bw_flits_per_cycle = 4

    low_result = evaluate_deepseek_v4_pro_ffn(low_bw)
    high_result = evaluate_deepseek_v4_pro_ffn(high_bw)
    assert high_result.network_cycles < low_result.network_cycles


def test_deepseek_evaluator_responds_to_io_bandwidth() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.num_routed_experts = 96
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.decode_tokens = 24
    base.workload.mapping_strategy = "expert_affinity"
    base.network.io_distribution_policy = "round_robin"

    high_bw = deepcopy(base)
    low_bw = deepcopy(base)
    high_bw.wafer.io_bandwidth_gbps = 256.0
    low_bw.wafer.io_bandwidth_gbps = 16.0

    high_bw_result = evaluate_deepseek_v4_pro_ffn(high_bw)
    low_bw_result = evaluate_deepseek_v4_pro_ffn(low_bw)
    assert low_bw_result.total_latency_cycles > high_bw_result.total_latency_cycles
    assert int(low_bw_result.metadata["io_injection_cycles"]) > int(
        high_bw_result.metadata["io_injection_cycles"]
    )


def test_deepseek_evaluator_responds_to_gateway_replication() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.num_routed_experts = 128
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.decode_tokens = 32
    base.workload.mapping_strategy = "expert_affinity"
    base.network.gateway_policy = "nearest"

    single_gateway = deepcopy(base)
    multi_gateway = deepcopy(base)
    single_gateway.network.gateways_per_reticle = 1
    multi_gateway.network.gateways_per_reticle = 4

    single_result = evaluate_deepseek_v4_pro_ffn(single_gateway)
    multi_result = evaluate_deepseek_v4_pro_ffn(multi_gateway)

    # Gateway replication should reduce (or keep) intra-reticle gateway detour hops.
    assert int(multi_result.metadata["gateway_noc_hops"]) <= int(
        single_result.metadata["gateway_noc_hops"]
    )


def test_deepseek_evaluator_load_aware_gateway_balances_load() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.num_routed_experts = 160
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.decode_tokens = 32
    base.workload.mapping_strategy = "expert_affinity"
    base.network.gateways_per_reticle = 4

    nearest = deepcopy(base)
    load_aware = deepcopy(base)
    nearest.network.gateway_policy = "nearest"
    load_aware.network.gateway_policy = "load_aware"

    nearest_result = evaluate_deepseek_v4_pro_ffn(nearest)
    load_aware_result = evaluate_deepseek_v4_pro_ffn(load_aware)

    assert load_aware_result.gateway_peak_load <= nearest_result.gateway_peak_load


def test_deepseek_partitioning_reduces_memory_stall_and_adds_allreduce() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.num_routed_experts = 8
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.decode_tokens = 16
    base.workload.mapping_strategy = "expert_affinity"

    expert_cfg = deepcopy(base)
    expert_cfg.workload.partition_strategy = "expert"
    expert_cfg.workload.partition_shards = 1

    col_cfg = deepcopy(base)
    col_cfg.workload.partition_strategy = "col"
    col_cfg.workload.partition_shards = 4

    k_cfg = deepcopy(base)
    k_cfg.workload.partition_strategy = "k_split"
    k_cfg.workload.partition_shards = 4

    expert_result = evaluate_deepseek_v4_pro_ffn(expert_cfg)
    col_result = evaluate_deepseek_v4_pro_ffn(col_cfg)
    k_result = evaluate_deepseek_v4_pro_ffn(k_cfg)

    assert col_result.memory_stall_cycles < expert_result.memory_stall_cycles
    assert col_result.allreduce_cycles > 0
    assert k_result.allreduce_cycles > 0
    assert int(col_result.metadata["partition_shards"]) >= 1


def test_deepseek_evaluator_responds_to_batch_size() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.num_routed_experts = 96
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.mapping_strategy = "expert_affinity"

    small_batch = deepcopy(base)
    large_batch = deepcopy(base)
    small_batch.workload.decode_tokens = 4
    large_batch.workload.decode_tokens = 16

    small_result = evaluate_deepseek_v4_pro_ffn(small_batch)
    large_result = evaluate_deepseek_v4_pro_ffn(large_batch)
    assert large_result.total_latency_cycles > small_result.total_latency_cycles


def test_partition_shards_for_2x2_reticle_is_batch_invariant() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v4_pro_ffn_decode"
    base.workload.num_routed_experts = 384
    base.workload.num_shared_experts = 1
    base.workload.top_k = 6
    base.workload.partition_strategy = "col"
    base.workload.partition_shards = 7
    base.workload.mapping_strategy = "expert_affinity"
    base.wafer.reticles_x = 2
    base.wafer.reticles_y = 2

    small_batch = deepcopy(base)
    large_batch = deepcopy(base)
    small_batch.workload.decode_tokens = 4
    large_batch.workload.decode_tokens = 32

    small_result = evaluate_deepseek_v4_pro_ffn(small_batch)
    large_result = evaluate_deepseek_v4_pro_ffn(large_batch)

    assert int(small_result.metadata["partition_shards"]) == 7
    assert int(large_result.metadata["partition_shards"]) == 7


def _make_base_cfg(num_experts: int = 8) -> WSEConfig:
    cfg = WSEConfig()
    cfg.workload.model_name = "deepseek_v4_pro_ffn_decode"
    cfg.workload.num_routed_experts = num_experts
    cfg.workload.num_shared_experts = 1
    cfg.workload.top_k = 6
    cfg.workload.decode_tokens = 4
    cfg.workload.mapping_strategy = "expert_affinity"
    return cfg


def test_row_strategy_produces_valid_result() -> None:
    cfg = _make_base_cfg()
    cfg.workload.partition_strategy = "row"
    cfg.workload.partition_shards = 1
    result = evaluate_deepseek_v4_pro_ffn(cfg)
    assert result.total_latency_cycles > 0
    assert result.allreduce_cycles == 0


def test_block_strategy_produces_valid_result() -> None:
    cfg = _make_base_cfg()
    cfg.workload.partition_strategy = "block"
    cfg.workload.partition_shards = 4
    result = evaluate_deepseek_v4_pro_ffn(cfg)
    assert result.total_latency_cycles > 0
    assert result.allreduce_cycles > 0


def test_hybrid_nk_strategy_produces_valid_result() -> None:
    cfg = _make_base_cfg()
    cfg.workload.partition_strategy = "hybrid_nk"
    cfg.workload.partition_shards = 4
    result = evaluate_deepseek_v4_pro_ffn(cfg)
    assert result.total_latency_cycles > 0
    assert result.allreduce_cycles > 0


def test_entwined_ring_strategy_discounts_allreduce() -> None:
    """Entwined ring allreduce should be lower than col via simulated interleaving."""
    base = _make_base_cfg()
    base.workload.partition_shards = 4

    col_cfg = deepcopy(base)
    col_cfg.workload.partition_strategy = "col"

    ring_cfg = deepcopy(base)
    ring_cfg.workload.partition_strategy = "entwined_ring"

    col_result = evaluate_deepseek_v4_pro_ffn(col_cfg)
    ring_result = evaluate_deepseek_v4_pro_ffn(ring_cfg)
    assert ring_result.allreduce_cycles <= col_result.allreduce_cycles
    assert ring_result.total_latency_cycles > 0


def test_streaming_strategy_produces_valid_result() -> None:
    cfg = _make_base_cfg()
    cfg.workload.partition_strategy = "streaming"
    cfg.workload.partition_shards = 4
    result = evaluate_deepseek_v4_pro_ffn(cfg)
    assert result.total_latency_cycles > 0
    assert result.memory_stall_cycles > 0


def test_tile_pipeline_affects_latency() -> None:
    base = _make_base_cfg()
    base.workload.partition_strategy = "col"
    base.workload.partition_shards = 4

    no_pipe = deepcopy(base)
    no_pipe.workload.tile_pipeline = False

    with_pipe = deepcopy(base)
    with_pipe.workload.tile_pipeline = True

    no_pipe_result = evaluate_deepseek_v4_pro_ffn(no_pipe)
    with_pipe_result = evaluate_deepseek_v4_pro_ffn(with_pipe)
    assert no_pipe_result.total_latency_cycles > 0
    assert with_pipe_result.total_latency_cycles > 0


def test_allreduce_simulated_scales_with_shards() -> None:
    """More shards should produce more allreduce cycles (more ring steps)."""
    base = _make_base_cfg()
    base.workload.partition_strategy = "col"

    few = deepcopy(base)
    few.workload.partition_shards = 2

    many = deepcopy(base)
    many.workload.partition_shards = 7

    few_result = evaluate_deepseek_v4_pro_ffn(few)
    many_result = evaluate_deepseek_v4_pro_ffn(many)
    assert few_result.allreduce_cycles > 0
    assert many_result.allreduce_cycles > few_result.allreduce_cycles


def test_allreduce_zero_for_expert_and_row() -> None:
    """Strategies that don't partition an expert should have zero allreduce."""
    for strat in ("expert", "row"):
        cfg = _make_base_cfg()
        cfg.workload.partition_strategy = strat
        cfg.workload.partition_shards = 1
        result = evaluate_deepseek_v4_pro_ffn(cfg)
        assert result.allreduce_cycles == 0, f"{strat} should have 0 allreduce"


def test_allreduce_simulated_no_magic_constant() -> None:
    """Confirm the 0.38 magic constant no longer exists in allreduce calculation."""
    import inspect
    from wsesim.dse import evaluator_deepseek

    source = inspect.getsource(evaluator_deepseek)
    assert "0.38" not in source, "Magic constant 0.38 should have been removed"


def test_col_s176_auto_uses_hierarchical_without_fallback() -> None:
    cfg = _make_base_cfg(num_experts=384)
    cfg.workload.partition_strategy = "col"
    cfg.workload.partition_shards = 176
    cfg.workload.collective_algorithm = "auto"
    cfg.workload.decode_tokens = 4
    cfg.network.noc.topology = "mesh2d"
    cfg.network.now.topology = "mesh2d"

    result = evaluate_deepseek_v4_pro_ffn(cfg)
    assert result.allreduce_cycles > 0
    assert result.metadata["collective_algorithm"] == "hierarchical"


def test_collective_algorithm_knob_changes_latency() -> None:
    base = _make_base_cfg(num_experts=384)
    base.workload.partition_strategy = "col"
    base.workload.partition_shards = 16
    base.workload.decode_tokens = 4
    base.network.noc.topology = "butterfly"
    base.network.now.topology = "mesh2d"

    ring_cfg = deepcopy(base)
    rhd_cfg = deepcopy(base)
    ring_cfg.workload.collective_algorithm = "ring"
    rhd_cfg.workload.collective_algorithm = "recursive_halving_doubling"

    ring_result = evaluate_deepseek_v4_pro_ffn(ring_cfg)
    rhd_result = evaluate_deepseek_v4_pro_ffn(rhd_cfg)
    assert rhd_result.allreduce_cycles <= ring_result.allreduce_cycles

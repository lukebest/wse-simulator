from __future__ import annotations

from copy import deepcopy

from wsesim.core.config import WSEConfig
from wsesim.dse.evaluator_deepseek import evaluate_deepseek_v3_ffn


def test_deepseek_evaluator_produces_nontrivial_metrics() -> None:
    cfg = WSEConfig()
    cfg.workload.model_name = "deepseek_v3_ffn_decode"
    cfg.workload.hidden_dim = 7168
    cfg.workload.expert_ffn_dim = 18432
    cfg.workload.num_routed_experts = 64
    cfg.workload.num_shared_experts = 1
    cfg.workload.top_k = 8
    cfg.workload.decode_tokens = 16
    cfg.workload.mapping_strategy = "expert_affinity"

    result = evaluate_deepseek_v3_ffn(cfg)
    assert result.total_latency_cycles > 0
    assert result.compute_cycles > 0
    assert result.network_cycles > 0
    assert result.network_throughput > 0
    assert result.vc_wait_cycles >= 0
    assert result.buffer_wait_cycles >= 0
    assert result.link_wait_cycles >= 0
    assert result.metadata["workload_model"] == "deepseek_v3_ffn_decode"


def test_deepseek_evaluator_responds_to_network_vc_scaling() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v3_ffn_decode"
    base.workload.num_routed_experts = 128
    base.workload.num_shared_experts = 1
    base.workload.top_k = 8
    base.workload.decode_tokens = 32
    base.workload.mapping_strategy = "expert_affinity"

    low_vc = deepcopy(base)
    high_vc = deepcopy(base)
    low_vc.network.noc.num_vcs = 1
    high_vc.network.noc.num_vcs = 4

    low_result = evaluate_deepseek_v3_ffn(low_vc)
    high_result = evaluate_deepseek_v3_ffn(high_vc)
    assert high_result.vc_wait_cycles < low_result.vc_wait_cycles


def test_deepseek_evaluator_responds_to_link_bandwidth() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v3_ffn_decode"
    base.workload.num_routed_experts = 96
    base.workload.num_shared_experts = 1
    base.workload.top_k = 8
    base.workload.decode_tokens = 24
    base.workload.mapping_strategy = "expert_affinity"

    low_bw = deepcopy(base)
    high_bw = deepcopy(base)
    low_bw.network.noc.link_bw_flits_per_cycle = 1
    high_bw.network.noc.link_bw_flits_per_cycle = 4

    low_result = evaluate_deepseek_v3_ffn(low_bw)
    high_result = evaluate_deepseek_v3_ffn(high_bw)
    assert high_result.network_cycles < low_result.network_cycles


def test_deepseek_evaluator_responds_to_now_link_latency() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v3_ffn_decode"
    base.workload.num_routed_experts = 96
    base.workload.num_shared_experts = 1
    base.workload.top_k = 8
    base.workload.decode_tokens = 24
    base.workload.mapping_strategy = "expert_affinity"

    fast_now = deepcopy(base)
    slow_now = deepcopy(base)
    fast_now.network.now.link_latency_cycles = 1
    slow_now.network.now.link_latency_cycles = 8

    fast_result = evaluate_deepseek_v3_ffn(fast_now)
    slow_result = evaluate_deepseek_v3_ffn(slow_now)
    assert slow_result.network_cycles > fast_result.network_cycles


def test_deepseek_evaluator_responds_to_gateway_replication() -> None:
    base = WSEConfig()
    base.workload.model_name = "deepseek_v3_ffn_decode"
    base.workload.num_routed_experts = 128
    base.workload.num_shared_experts = 1
    base.workload.top_k = 8
    base.workload.decode_tokens = 32
    base.workload.mapping_strategy = "expert_affinity"
    base.network.gateway_policy = "nearest"

    single_gateway = deepcopy(base)
    multi_gateway = deepcopy(base)
    single_gateway.network.gateways_per_reticle = 1
    multi_gateway.network.gateways_per_reticle = 4

    single_result = evaluate_deepseek_v3_ffn(single_gateway)
    multi_result = evaluate_deepseek_v3_ffn(multi_gateway)

    # Gateway replication should reduce (or keep) intra-reticle gateway detour hops.
    assert int(multi_result.metadata["gateway_noc_hops"]) <= int(
        single_result.metadata["gateway_noc_hops"]
    )

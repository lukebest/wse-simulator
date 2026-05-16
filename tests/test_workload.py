from __future__ import annotations

from wsesim.workload.generator import (
    DeepSeekV3FFNProfile,
    generate_deepseek_v3_decode_ffn_workload,
    generate_moe_decode_ffn_workload,
)
from wsesim.workload.mapper import ExpertAffinityMapping, NearestNeighborMapping
from wsesim.workload.partition.expert import ExpertPartition


def test_generate_moe_decode_workload() -> None:
    workload = generate_moe_decode_ffn_workload(
        model_name="mixtral_8x7b",
        hidden_dim=4096,
        expert_ffn_dim=14336,
        num_experts=8,
        top_k=2,
        decode_tokens=4,
    )
    assert workload.ops[0].name == "router_gate_proj"
    assert workload.ops[-1].name == "token_combine"
    assert len(workload.ops) == 2 + 2 * 8 + 1


def test_partition_and_mapping() -> None:
    workload = generate_moe_decode_ffn_workload(
        model_name="mixtral_8x7b",
        hidden_dim=1024,
        expert_ffn_dim=2048,
        num_experts=4,
        top_k=2,
        decode_tokens=2,
    )
    partitioner = ExpertPartition()
    tasks = {op.name: partitioner.partition(op, shards=2) for op in workload.ops}
    mapping = NearestNeighborMapping().map(workload, tasks, alive_cores=[0, 1, 2, 3])
    assert "router_gate_proj" in mapping.assignments
    assert len(mapping.core_tasks) > 0


def test_generate_deepseek_v3_decode_ffn_workload() -> None:
    profile = DeepSeekV3FFNProfile(
        num_routed_experts=16,
        num_shared_experts=1,
        top_k=4,
        decode_tokens=8,
    )
    workload = generate_deepseek_v3_decode_ffn_workload(profile)
    assert workload.model_name == "deepseek_v3_ffn_decode"
    assert workload.ops[0].name == "deepseek_v3_router_gate_proj"
    assert workload.ops[-1].name == "deepseek_v3_token_combine"
    assert int(workload.metadata["active_routed_experts"]) > 0
    assert int(workload.metadata["active_routed_experts"]) <= profile.num_routed_experts


def test_deepseek_expert_affinity_mapping_keeps_expert_locality() -> None:
    profile = DeepSeekV3FFNProfile(
        num_routed_experts=12,
        num_shared_experts=1,
        top_k=2,
        decode_tokens=6,
    )
    workload = generate_deepseek_v3_decode_ffn_workload(profile)
    partitioner = ExpertPartition()
    tasks = {op.name: partitioner.partition(op, shards=1) for op in workload.ops}
    mapping = ExpertAffinityMapping().map(workload, tasks, alive_cores=list(range(8)))

    target_expert = 3
    up_name = f"deepseek_v3_expert_routed_{target_expert}_up_proj"
    down_name = f"deepseek_v3_expert_routed_{target_expert}_down_proj"
    if up_name in mapping.assignments and down_name in mapping.assignments:
        assert mapping.assignments[up_name][0] == mapping.assignments[down_name][0]

    assert mapping.assignments["deepseek_v3_router_gate_proj"][0] == 0
    assert mapping.assignments["deepseek_v3_token_dispatch"][0] == 0

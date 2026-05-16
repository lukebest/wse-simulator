from __future__ import annotations

from wsesim.workload.generator import generate_moe_decode_ffn_workload
from wsesim.workload.mapper import NearestNeighborMapping
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
    assert len(workload.token_routes) == 4
    assert all(len(route.selected_experts) == 2 for route in workload.token_routes)
    assert all(abs(sum(route.gate_scores) - 1.0) < 1e-6 for route in workload.token_routes)

    expert_counts = {
        idx: sum(idx in route.selected_experts for route in workload.token_routes)
        for idx in range(8)
    }
    for idx in range(8):
        gate_op = next(op for op in workload.ops if op.name == f"expert_{idx}_gate_proj")
        assert gate_op.m == expert_counts[idx]


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
    assert len(mapping.token_home_cores) == workload.metadata["decode_tokens"]
    expected_transfers = workload.metadata["decode_tokens"] * workload.metadata["top_k"]
    assert len(mapping.token_dispatch_transfers) == expected_transfers
    assert len(mapping.token_combine_transfers) == expected_transfers
    assert all(t.collective == "moe_dispatch" for t in mapping.token_dispatch_transfers)
    assert all(t.collective == "moe_combine" for t in mapping.token_combine_transfers)

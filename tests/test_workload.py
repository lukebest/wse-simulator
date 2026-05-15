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

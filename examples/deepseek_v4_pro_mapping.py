"""Generate DeepSeek-V4-Pro FFN workload and map it onto WSE cores."""

from __future__ import annotations

from collections import Counter

from wsesim.workload.generator import (
    DeepSeekV4ProFFNProfile,
    generate_deepseek_v4_pro_decode_ffn_workload,
)
from wsesim.workload.mapper import ExpertAffinityMapping
from wsesim.workload.partition.expert import ExpertPartition


def main() -> None:
    profile = DeepSeekV4ProFFNProfile(
        hidden_dim=7168,
        expert_ffn_dim=3072,
        num_routed_experts=384,
        num_shared_experts=1,
        top_k=6,
        decode_tokens=32,
        routing_skew_alpha=1.2,
        capacity_factor=1.25,
    )
    workload = generate_deepseek_v4_pro_decode_ffn_workload(profile)

    partitioner = ExpertPartition()
    tasks = {op.name: partitioner.partition(op, shards=1) for op in workload.ops}

    alive_cores = list(range(128))  # Example WSE slice
    mapping = ExpertAffinityMapping().map(workload, tasks, alive_cores=alive_cores)

    op_count = len(workload.ops)
    active_routed = int(workload.metadata["active_routed_experts"])
    core_load = Counter({core: len(core_tasks) for core, core_tasks in mapping.core_tasks.items()})

    print("model:", workload.model_name)
    print("ops:", op_count)
    print("active_routed_experts:", active_routed)
    print("mapped_cores:", len(mapping.core_tasks))
    print("top_busy_cores:", core_load.most_common(10))
    print("sample_mapping:")
    for op_name in list(mapping.assignments.keys())[:10]:
        print(f"  {op_name} -> {mapping.assignments[op_name]}")


if __name__ == "__main__":
    main()

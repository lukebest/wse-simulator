"""Workload-to-core mapping strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from wsesim.workload.collective import CollectiveOps, P2PTransfer
from wsesim.workload.ops import LLMWorkload
from wsesim.workload.partition.base import TileTask


@dataclass(slots=True)
class Mapping:
    assignments: dict[str, list[int]] = field(default_factory=dict)
    core_tasks: dict[int, list[TileTask]] = field(default_factory=dict)
    token_home_cores: dict[int, int] = field(default_factory=dict)
    expert_cores: dict[int, int] = field(default_factory=dict)
    token_dispatch_transfers: list[P2PTransfer] = field(default_factory=list)
    token_combine_transfers: list[P2PTransfer] = field(default_factory=list)


class MappingStrategy(ABC):
    @abstractmethod
    def map(self, workload: LLMWorkload, tasks: dict[str, list[TileTask]], alive_cores: list[int]) -> Mapping:
        raise NotImplementedError


class NearestNeighborMapping(MappingStrategy):
    def map(self, workload: LLMWorkload, tasks: dict[str, list[TileTask]], alive_cores: list[int]) -> Mapping:
        if not alive_cores:
            raise ValueError("No alive cores available for mapping.")

        mapping = Mapping()
        cursor = 0
        for op in workload.ops:
            op_tasks = tasks.get(op.name, [])
            mapped_cores: list[int] = []
            for task in op_tasks:
                core = alive_cores[cursor % len(alive_cores)]
                mapped_cores.append(core)
                mapping.core_tasks.setdefault(core, []).append(task)
                cursor += 1
            mapping.assignments[op.name] = mapped_cores

        _attach_token_level_plan(workload, mapping, alive_cores)
        return mapping


class ExpertLocalityMapping(MappingStrategy):
    def map(self, workload: LLMWorkload, tasks: dict[str, list[TileTask]], alive_cores: list[int]) -> Mapping:
        # In this initial version, keep behavior deterministic and contiguous.
        if not alive_cores:
            raise ValueError("No alive cores available for mapping.")
        mapping = Mapping()
        start = 0
        for op in workload.ops:
            op_tasks = tasks.get(op.name, [])
            count = len(op_tasks)
            chunk = alive_cores[start : start + count] or alive_cores
            mapped_cores: list[int] = []
            for idx, task in enumerate(op_tasks):
                core = chunk[idx % len(chunk)]
                mapped_cores.append(core)
                mapping.core_tasks.setdefault(core, []).append(task)
            mapping.assignments[op.name] = mapped_cores
            start = (start + count) % len(alive_cores)
        _attach_token_level_plan(workload, mapping, alive_cores)
        return mapping


def _attach_token_level_plan(
    workload: LLMWorkload,
    mapping: Mapping,
    alive_cores: list[int],
) -> None:
    if not workload.token_routes:
        return

    token_home_cores = {
        route.token_id: alive_cores[route.token_id % len(alive_cores)]
        for route in workload.token_routes
    }
    expert_cores = {}
    num_experts = int(workload.metadata.get("num_experts", 0))
    for expert_id in range(num_experts):
        op_name = f"expert_{expert_id}_gate_proj"
        assigned = mapping.assignments.get(op_name, [])
        if assigned:
            expert_cores[expert_id] = assigned[0]
        else:
            expert_cores[expert_id] = alive_cores[expert_id % len(alive_cores)]

    hidden_dim = int(workload.metadata.get("hidden_dim", 4096))
    token_bytes = hidden_dim * 2  # fp16/bf16
    mapping.token_home_cores = token_home_cores
    mapping.expert_cores = expert_cores
    mapping.token_dispatch_transfers = CollectiveOps.moe_dispatch(
        token_routes=workload.token_routes,
        token_home_cores=token_home_cores,
        expert_cores=expert_cores,
        token_bytes=token_bytes,
    )
    mapping.token_combine_transfers = CollectiveOps.moe_combine(
        token_routes=workload.token_routes,
        token_home_cores=token_home_cores,
        expert_cores=expert_cores,
        token_bytes=token_bytes,
    )

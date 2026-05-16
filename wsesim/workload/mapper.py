"""Workload-to-core mapping strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from wsesim.workload.ops import LLMWorkload
from wsesim.workload.partition.base import TileTask


@dataclass(slots=True)
class Mapping:
    assignments: dict[str, list[int]] = field(default_factory=dict)
    core_tasks: dict[int, list[TileTask]] = field(default_factory=dict)


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
        return mapping


class ExpertAffinityMapping(MappingStrategy):
    """Map routed experts consistently to affinity cores.

    - Router/dispatch/combine ops are anchored on coordinator cores.
    - Expert ops with the same expert_id always map to the same core.
    """

    def map(self, workload: LLMWorkload, tasks: dict[str, list[TileTask]], alive_cores: list[int]) -> Mapping:
        if not alive_cores:
            raise ValueError("No alive cores available for mapping.")

        mapping = Mapping()
        coordinator = alive_cores[0]
        lookup = {op.name: op for op in workload.ops}

        for op in workload.ops:
            op_tasks = tasks.get(op.name, [])
            mapped_cores: list[int] = []
            op_meta = lookup[op.name]

            if op_meta.op_type in {"router", "dispatch", "combine"}:
                for task in op_tasks:
                    mapped_cores.append(coordinator)
                    mapping.core_tasks.setdefault(coordinator, []).append(task)
                mapping.assignments[op.name] = mapped_cores
                continue

            if op_meta.expert_id is not None:
                if op_meta.expert_kind == "shared":
                    core = alive_cores[(op_meta.expert_id + 1) % len(alive_cores)]
                else:
                    core = alive_cores[op_meta.expert_id % len(alive_cores)]
                for task in op_tasks:
                    mapped_cores.append(core)
                    mapping.core_tasks.setdefault(core, []).append(task)
                mapping.assignments[op.name] = mapped_cores
                continue

            # Fallback for unknown ops.
            for idx, task in enumerate(op_tasks):
                core = alive_cores[idx % len(alive_cores)]
                mapped_cores.append(core)
                mapping.core_tasks.setdefault(core, []).append(task)
            mapping.assignments[op.name] = mapped_cores

        return mapping

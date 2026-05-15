"""Partition strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from wsesim.workload.ops import GEMMOp


@dataclass(slots=True)
class TileTask:
    op_name: str
    shard_id: int
    m: int
    n: int
    k: int


class PartitionStrategy(ABC):
    @abstractmethod
    def partition(self, op: GEMMOp, shards: int) -> list[TileTask]:
        raise NotImplementedError

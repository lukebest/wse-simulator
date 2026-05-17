"""Hybrid N-K partitioning: K-split for W1/W3, col-split for W2."""

from __future__ import annotations

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.base import PartitionStrategy, TileTask
from wsesim.workload.partition.col import ColPartition
from wsesim.workload.partition.k_split import KPartition


class HybridNKPartition(PartitionStrategy):
    """Choose the best split dimension per op based on K vs N ratio."""

    def __init__(self) -> None:
        self._k_split = KPartition()
        self._col = ColPartition()

    def partition(self, op: GEMMOp, shards: int) -> list[TileTask]:
        if op.k >= op.n:
            return self._k_split.partition(op, shards)
        return self._col.partition(op, shards)

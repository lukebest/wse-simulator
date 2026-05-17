"""Entwined ring mapping partition (MoEntwine HPCA 2026).

Tile generation is identical to col-split; the communication advantage
is modeled in the evaluator via a ring-prefetch discount factor.
"""

from __future__ import annotations

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.base import PartitionStrategy, TileTask
from wsesim.workload.partition.col import ColPartition


class EntwinedRingPartition(PartitionStrategy):
    def __init__(self) -> None:
        self._col = ColPartition()

    def partition(self, op: GEMMOp, shards: int) -> list[TileTask]:
        return self._col.partition(op, shards)

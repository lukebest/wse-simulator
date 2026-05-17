"""Expert streaming partition (FSE-DP).

Weights are streamed through cores in K-dimension chunks.
Tile shapes match K-split; the pipelined compute-memory overlap
is modeled in the evaluator.
"""

from __future__ import annotations

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.base import PartitionStrategy, TileTask
from wsesim.workload.partition.k_split import KPartition


class StreamingPartition(PartitionStrategy):
    def __init__(self) -> None:
        self._k_split = KPartition()

    def partition(self, op: GEMMOp, shards: int) -> list[TileTask]:
        return self._k_split.partition(op, shards)

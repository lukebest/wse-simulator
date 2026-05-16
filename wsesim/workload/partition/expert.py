"""MoE expert-oriented partitioning."""

from __future__ import annotations

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.base import PartitionStrategy, TileTask


class ExpertPartition(PartitionStrategy):
    def partition(self, op: GEMMOp, shards: int) -> list[TileTask]:
        # For expert ops, map one shard per expert where possible.
        if op.m <= 0:
            return []
        tasks: list[TileTask] = []
        for shard in range(max(1, shards)):
            tasks.append(TileTask(op_name=op.name, shard_id=shard, m=op.m, n=op.n, k=op.k))
        return tasks

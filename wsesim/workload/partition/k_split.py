"""K-dimension partitioning."""

from __future__ import annotations

from math import ceil

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.base import PartitionStrategy, TileTask


class KPartition(PartitionStrategy):
    def partition(self, op: GEMMOp, shards: int) -> list[TileTask]:
        k_per_shard = ceil(op.k / max(shards, 1))
        tasks: list[TileTask] = []
        for shard in range(max(shards, 1)):
            k_shard = max(0, min(k_per_shard, op.k - shard * k_per_shard))
            if k_shard == 0:
                continue
            tasks.append(TileTask(op_name=op.name, shard_id=shard, m=op.m, n=op.n, k=k_shard))
        return tasks

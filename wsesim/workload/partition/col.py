"""Column-wise partitioning."""

from __future__ import annotations

from math import ceil

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.base import PartitionStrategy, TileTask


class ColPartition(PartitionStrategy):
    def partition(self, op: GEMMOp, shards: int) -> list[TileTask]:
        cols_per_shard = ceil(op.n / max(shards, 1))
        tasks: list[TileTask] = []
        for shard in range(shards):
            n_shard = max(0, min(cols_per_shard, op.n - shard * cols_per_shard))
            if n_shard == 0:
                continue
            tasks.append(TileTask(op_name=op.name, shard_id=shard, m=op.m, n=n_shard, k=op.k))
        return tasks

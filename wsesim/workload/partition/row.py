"""Row-wise partitioning."""

from __future__ import annotations

from math import ceil

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.base import PartitionStrategy, TileTask


class RowPartition(PartitionStrategy):
    def partition(self, op: GEMMOp, shards: int) -> list[TileTask]:
        rows_per_shard = ceil(op.m / max(shards, 1))
        tasks: list[TileTask] = []
        for shard in range(shards):
            m_shard = max(0, min(rows_per_shard, op.m - shard * rows_per_shard))
            if m_shard == 0:
                continue
            tasks.append(TileTask(op_name=op.name, shard_id=shard, m=m_shard, n=op.n, k=op.k))
        return tasks

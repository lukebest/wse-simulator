"""2D block partitioning."""

from __future__ import annotations

from math import ceil, isqrt

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.base import PartitionStrategy, TileTask


class BlockPartition(PartitionStrategy):
    def partition(self, op: GEMMOp, shards: int) -> list[TileTask]:
        side = max(1, isqrt(max(shards, 1)))
        block_rows = ceil(op.m / side)
        block_cols = ceil(op.n / side)
        tasks: list[TileTask] = []
        shard_id = 0
        for r in range(side):
            for c in range(side):
                if shard_id >= shards:
                    break
                m_shard = max(0, min(block_rows, op.m - r * block_rows))
                n_shard = max(0, min(block_cols, op.n - c * block_cols))
                if m_shard == 0 or n_shard == 0:
                    continue
                tasks.append(
                    TileTask(op_name=op.name, shard_id=shard_id, m=m_shard, n=n_shard, k=op.k)
                )
                shard_id += 1
        return tasks

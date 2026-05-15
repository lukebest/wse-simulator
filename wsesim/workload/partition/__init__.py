"""Partition strategies."""

from wsesim.workload.partition.base import PartitionStrategy, TileTask
from wsesim.workload.partition.block import BlockPartition
from wsesim.workload.partition.col import ColPartition
from wsesim.workload.partition.expert import ExpertPartition
from wsesim.workload.partition.row import RowPartition

__all__ = [
    "PartitionStrategy",
    "TileTask",
    "RowPartition",
    "ColPartition",
    "BlockPartition",
    "ExpertPartition",
]

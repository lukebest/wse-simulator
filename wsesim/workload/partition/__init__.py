"""Partition strategies."""

from wsesim.workload.partition.base import PartitionStrategy, TileTask
from wsesim.workload.partition.block import BlockPartition
from wsesim.workload.partition.col import ColPartition
from wsesim.workload.partition.entwined_ring import EntwinedRingPartition
from wsesim.workload.partition.expert import ExpertPartition
from wsesim.workload.partition.hybrid_nk import HybridNKPartition
from wsesim.workload.partition.k_split import KPartition
from wsesim.workload.partition.row import RowPartition
from wsesim.workload.partition.streaming import StreamingPartition

__all__ = [
    "PartitionStrategy",
    "TileTask",
    "RowPartition",
    "ColPartition",
    "BlockPartition",
    "ExpertPartition",
    "KPartition",
    "HybridNKPartition",
    "EntwinedRingPartition",
    "StreamingPartition",
]

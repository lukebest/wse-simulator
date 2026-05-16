from __future__ import annotations

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.k_split import KPartition


def test_k_partition_splits_k_dimension() -> None:
    op = GEMMOp(name="op", m=16, n=32, k=64)
    parts = KPartition().partition(op, shards=4)
    assert len(parts) == 4
    assert all(part.m == 16 for part in parts)
    assert all(part.n == 32 for part in parts)
    assert [part.k for part in parts] == [16, 16, 16, 16]


def test_k_partition_handles_remainder() -> None:
    op = GEMMOp(name="op", m=8, n=12, k=10)
    parts = KPartition().partition(op, shards=3)
    assert len(parts) == 3
    assert [part.k for part in parts] == [4, 4, 2]


def test_k_partition_minimum_one_shard() -> None:
    op = GEMMOp(name="op", m=4, n=4, k=7)
    parts = KPartition().partition(op, shards=0)
    assert len(parts) == 1
    assert parts[0].k == 7

from __future__ import annotations

from wsesim.workload.ops import GEMMOp
from wsesim.workload.partition.entwined_ring import EntwinedRingPartition
from wsesim.workload.partition.hybrid_nk import HybridNKPartition
from wsesim.workload.partition.k_split import KPartition
from wsesim.workload.partition.streaming import StreamingPartition


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


def test_hybrid_nk_uses_k_split_when_k_larger() -> None:
    op = GEMMOp(name="w1", m=4, n=3072, k=7168)
    parts = HybridNKPartition().partition(op, shards=4)
    assert len(parts) == 4
    assert all(p.n == 3072 for p in parts)
    assert all(p.k <= 1792 for p in parts)


def test_hybrid_nk_uses_col_when_n_larger() -> None:
    op = GEMMOp(name="w2", m=4, n=7168, k=3072)
    parts = HybridNKPartition().partition(op, shards=4)
    assert len(parts) == 4
    assert all(p.k == 3072 for p in parts)
    assert all(p.n <= 1792 for p in parts)


def test_entwined_ring_same_as_col() -> None:
    op = GEMMOp(name="op", m=4, n=7168, k=3072)
    from wsesim.workload.partition.col import ColPartition
    col_parts = ColPartition().partition(op, shards=4)
    ring_parts = EntwinedRingPartition().partition(op, shards=4)
    assert len(col_parts) == len(ring_parts)
    for c, r in zip(col_parts, ring_parts):
        assert c.m == r.m and c.n == r.n and c.k == r.k


def test_streaming_same_tiles_as_k_split() -> None:
    op = GEMMOp(name="op", m=4, n=3072, k=7168)
    k_parts = KPartition().partition(op, shards=4)
    s_parts = StreamingPartition().partition(op, shards=4)
    assert len(k_parts) == len(s_parts)
    for kp, sp in zip(k_parts, s_parts):
        assert kp.m == sp.m and kp.n == sp.n and kp.k == sp.k

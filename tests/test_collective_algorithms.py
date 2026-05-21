from __future__ import annotations

from wsesim.network.collective import (
    generate_collective_traffic,
    select_collective_algorithm,
)


def test_collective_algorithms_generate_valid_traffic_small_s() -> None:
    nodes = [0, 1, 2, 3]
    payload = 4096
    num_experts = 2
    for algorithm in (
        "ring",
        "recursive_halving_doubling",
        "2d_ring",
        "direct_allgather",
        "hierarchical",
    ):
        traffic = generate_collective_traffic(
            algorithm=algorithm,
            participating_nodes_global=nodes,
            cores_per_reticle=4,
            payload_bytes_per_expert=payload,
            num_experts=num_experts,
            topology_hint={"rows": 2, "cols": 2},
        )
        assert traffic, f"{algorithm} should generate traffic"
        total_bytes = 0
        for pkt in traffic:
            assert pkt["src_core"] in nodes
            assert pkt["dst_core"] in nodes
            assert int(pkt["size_bytes"]) > 0
            total_bytes += int(pkt["size_bytes"])
        assert total_bytes > 0


def test_select_collective_algorithm_rules() -> None:
    assert (
        select_collective_algorithm(
            partition_strategy="col",
            noc_topology="mesh2d",
            now_topology="mesh2d",
            shards=176,
            cores_per_reticle=44,
            reticle_count=4,
        )
        == "hierarchical"
    )
    assert (
        select_collective_algorithm(
            partition_strategy="col",
            noc_topology="mesh2d",
            now_topology="mesh2d",
            shards=8,
            cores_per_reticle=44,
            reticle_count=4,
        )
        == "direct_allgather"
    )
    assert (
        select_collective_algorithm(
            partition_strategy="k_split",
            noc_topology="butterfly",
            now_topology="mesh2d",
            shards=16,
            cores_per_reticle=44,
            reticle_count=4,
        )
        == "recursive_halving_doubling"
    )
    assert (
        select_collective_algorithm(
            partition_strategy="col",
            noc_topology="mesh2d",
            now_topology="mesh2d",
            shards=12,
            cores_per_reticle=44,
            reticle_count=4,
        )
        == "2d_ring"
    )
    assert (
        select_collective_algorithm(
            partition_strategy="col",
            noc_topology="supermesh_bi",
            now_topology="mesh2d",
            shards=13,
            cores_per_reticle=44,
            reticle_count=4,
        )
        == "ring"
    )


def test_hierarchical_traffic_contains_cross_reticle_packets() -> None:
    nodes = list(range(176))
    traffic = generate_collective_traffic(
        algorithm="hierarchical",
        participating_nodes_global=nodes,
        cores_per_reticle=44,
        payload_bytes_per_expert=8192,
        num_experts=1,
    )
    assert traffic
    assert any(
        int(pkt["src_core"]) // 44 != int(pkt["dst_core"]) // 44 for pkt in traffic
    )

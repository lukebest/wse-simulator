from __future__ import annotations

from wsesim.network.topology.butterfly import Butterfly
from wsesim.network.topology.flat_butterfly import FlatButterfly
from wsesim.network.topology.supermesh_alter import SuperMeshAlter
from wsesim.network.topology.supermesh_bi import SuperMeshBi


def test_flat_butterfly_6x8_grouping_links() -> None:
    topo = FlatButterfly()
    graph = topo.build(48)

    # Node 0 is in first 2x4 block: [0,1,2,3,8,9,10,11]
    for peer in [1, 2, 3, 8, 9, 10, 11]:
        assert peer in graph[0]

    # Lane-0 links to all other blocks.
    for lane_peer in [4, 16, 20, 32, 36]:
        assert lane_peer in graph[0]


def test_supermesh_bi_enhanced_edges_full_perimeter() -> None:
    topo = SuperMeshBi(rows=6, cols=8)
    edges = topo.enhanced_edges(48)

    assert (0, 1) in edges
    assert (6, 7) in edges
    assert (40, 41) in edges
    assert (46, 47) in edges
    assert (0, 8) in edges
    assert (8, 16) in edges
    assert (32, 40) in edges
    assert (7, 15) in edges
    assert (31, 39) in edges
    assert (39, 47) in edges


def test_supermesh_alter_enhanced_edges_alternating_perimeter() -> None:
    topo = SuperMeshAlter(rows=6, cols=8)
    edges = topo.enhanced_edges(48)

    # Horizontal alternating.
    assert (0, 1) in edges
    assert (2, 3) in edges
    assert (1, 2) not in edges
    assert (3, 4) not in edges

    # Vertical alternating.
    assert (0, 8) in edges
    assert (16, 24) in edges
    assert (32, 40) in edges
    assert (8, 16) not in edges
    assert (24, 32) not in edges


def test_butterfly_build_has_exchange_paths() -> None:
    topo = Butterfly(rows=6, cols=8)
    graph = topo.build(48)

    # Node (row0,col0)=0 connects to straight (0,1)=1 and exchange peer.
    assert 1 in graph[0]
    assert len(graph[0]) >= 2

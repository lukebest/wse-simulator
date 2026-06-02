from __future__ import annotations

import simpy

from wsesim.network.collective import generate_collective_traffic, _resolve_groups_by_dimension
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.topology.rect_flat_butterfly import RectFlatButterfly
from wsesim.network.topology.restricted_hypercube_fb import RestrictedHypercubeFB


def test_rect_flat_butterfly_6x8_coloring() -> None:
    topo = RectFlatButterfly(rows=6, cols=8)
    plan = topo.coloring()
    assert plan.C == 16
    assert len(topo.logical_links()) == 576


def test_restricted_hypercube_6x8_coloring() -> None:
    topo = RestrictedHypercubeFB(n=6, parent_rows=8, parent_cols=8, keep_rows=6, keep_cols=8)
    plan = topo.coloring()
    assert plan.C == 5
    assert len(topo.logical_links()) == 256


def test_restricted_hypercube_nd_groups_are_pairs() -> None:
    topo = RestrictedHypercubeFB(n=6, parent_rows=8, parent_cols=8, keep_rows=6, keep_cols=8)
    nodes = topo.node_ids()
    hint = {
        "k": 2,
        "n": 6,
        "hypercube_coords": {node: topo.to_coords(node) for node in nodes},
    }
    groups = _resolve_groups_by_dimension(nodes, hint)
    assert len(groups) == 6
    for dim_groups in groups.values():
        for group in dim_groups:
            assert 1 <= len(group) <= 2


def test_rect_6x8_nd_delivery() -> None:
    topo = RectFlatButterfly(rows=6, cols=8)
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=topo,
        routing=TDMFlatButterflyRouting(topology=topo),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=48,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
    )
    traffic = generate_collective_traffic(
        algorithm="nd_dimension_exchange_allgather",
        participating_nodes_global=list(range(48)),
        cores_per_reticle=48,
        payload_bytes_per_expert=1024,
        num_experts=1,
        topology_hint={"k_dims": [8, 6], "rows": 6, "cols": 8},
    )
    for item in traffic[:8]:
        env.process(
            net.send_packet(
                Packet(
                    src=int(item["src_core"]),
                    dst=int(item["dst_core"]),
                    size_bytes=int(item["size_bytes"]),
                    payload_type=str(item["payload"]),
                )
            )
        )
    env.run()
    assert net.stats.packets_sent == 8

from __future__ import annotations

import simpy

from wsesim.network.collective import generate_collective_traffic
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly


def test_tdm_flat_butterfly_4x4_roundtrip() -> None:
    for k, n in ((4, 2), (2, 4)):
        topo = TDMFlatButterfly(k=k, n=n, rows=4, cols=4)
        for node in range(16):
            assert topo.to_node(topo.to_coords(node)) == node


def test_tdm_flat_butterfly_4x4_coloring_conflict_free() -> None:
    for k, n in ((4, 2), (2, 4)):
        topo = TDMFlatButterfly(k=k, n=n, rows=4, cols=4)
        plan = topo.coloring()
        assert plan.C >= plan.color_lower_bound
        for color in range(plan.C):
            used: set[tuple[int, int]] = set()
            for edge, owners in plan.link_active.items():
                if owners[color] is None:
                    continue
                assert edge not in used
                used.add(edge)


def test_tdm_flat_butterfly_4x4_network_runs() -> None:
    env = simpy.Environment()
    topo = TDMFlatButterfly(k=4, n=2, rows=4, cols=4)
    net = UnifiedNetwork(
        env=env,
        topology=topo,
        routing=TDMFlatButterflyRouting(topology=topo),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=16,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
    )
    env.process(net.send_packet(Packet(src=0, dst=15, size_bytes=128, payload_type="activation")))
    env.run()
    assert net.stats.packets_sent == 1
    assert net.stats.max_packet_latency > 0


def test_tdm_flat_butterfly_4x4_route_colors() -> None:
    topo = TDMFlatButterfly(k=2, n=4, rows=4, cols=4)
    assert topo.route_color(0, 5) == 1
    assert topo.route_color(0, 6) is None
    min_c, color = topo.min_route_color(0, 6)
    assert min_c == 2
    assert color == 0


def test_tdm_flat_butterfly_4x4_collective_traffic() -> None:
    traffic = generate_collective_traffic(
        algorithm="nd_dimension_exchange_allreduce",
        participating_nodes_global=list(range(16)),
        cores_per_reticle=16,
        payload_bytes_per_expert=1024,
        num_experts=1,
        topology_hint={"k": 4, "n": 2},
    )
    assert traffic
    assert all(0 <= int(pkt["src_core"]) < 16 for pkt in traffic)
    assert all(0 <= int(pkt["dst_core"]) < 16 for pkt in traffic)

from __future__ import annotations

import simpy

from wsesim.network.collective import generate_collective_traffic
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.tdm_link import TDMLink
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly


def test_tdm_flat_butterfly_coords_roundtrip() -> None:
    for k, n in ((8, 2), (4, 3), (2, 6)):
        topo = TDMFlatButterfly(k=k, n=n, rows=8, cols=8)
        for node in range(64):
            coords = topo.to_coords(node)
            assert len(coords) == n
            assert topo.to_node(coords) == node


def test_tdm_flat_butterfly_coloring_conflict_free() -> None:
    for k, n in ((8, 2), (4, 3), (2, 6)):
        topo = TDMFlatButterfly(k=k, n=n, rows=8, cols=8)
        plan = topo.coloring()
        assert plan.C >= plan.color_lower_bound

        logical_links = {(u, v) for u, v, _ in topo.logical_links()}
        assert set(plan.color_of_logical.keys()) == logical_links

        for color in range(plan.C):
            used_edges: set[tuple[int, int]] = set()
            for edge, owners in plan.link_active.items():
                active = owners[color]
                if active is None:
                    continue
                assert edge not in used_edges
                used_edges.add(edge)


def test_tdm_link_does_not_spin_for_color_slot() -> None:
    """Color gating is router-only; link should not add TDM slot spin."""
    env = simpy.Environment()
    link = TDMLink(
        env=env,
        src=0,
        dst=1,
        bandwidth_flits_per_cycle=1,
        latency_cycles=1,
        period=4,
        active_logical_per_color=[None, (0, 1), None, None],
    )
    env.process(link.transfer(1, flit_color=1, logical_link=(0, 1)))
    env.run()
    assert env.now == 2
    assert link.total_wait_cycles == 0


def test_tdm_network_packet_delivery_and_wait_cycles() -> None:
    env = simpy.Environment()
    topo = TDMFlatButterfly(k=8, n=2, rows=8, cols=8)
    net = UnifiedNetwork(
        env=env,
        topology=topo,
        routing=TDMFlatButterflyRouting(topology=topo),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=64,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
    )
    env.process(net.send_packet(Packet(src=0, dst=63, size_bytes=128, payload_type="activation")))
    env.run()
    assert net.stats.packets_sent == 1
    assert net.stats.max_packet_latency > 0
    assert net.stats.color_buffer_wait_cycles > 0


def test_k2n6_allreduce_not_far_worse_than_k8n2() -> None:
    def run_collective(k: int, n: int) -> float:
        env = simpy.Environment()
        topo = TDMFlatButterfly(k=k, n=n, rows=8, cols=8)
        net = UnifiedNetwork(
            env=env,
            topology=topo,
            routing=TDMFlatButterflyRouting(topology=topo),
            flow_control=CreditBasedVCFlowControl(),
            num_nodes=64,
            link_bw_flits_per_cycle=1,
            link_latency_cycles=1,
            num_vcs=2,
            buffer_depth=8,
        )
        traffic = generate_collective_traffic(
            algorithm="nd_dimension_exchange_allreduce",
            participating_nodes_global=list(range(64)),
            cores_per_reticle=64,
            payload_bytes_per_expert=1024,
            num_experts=1,
            topology_hint={"k": k, "n": n},
        )
        for item in traffic:
            delay = int(item.get("delay_cycles", 0))

            def _inject(
                sim_env: simpy.Environment, delay_cycles: int, src: int, dst: int, size_bytes: int, payload: str
            ):
                if delay_cycles > 0:
                    yield sim_env.timeout(delay_cycles)
                yield sim_env.process(
                    net.send_packet(
                        Packet(src=src, dst=dst, size_bytes=size_bytes, payload_type=payload)
                    )
                )

            env.process(
                _inject(
                    env,
                    delay,
                    int(item["src_core"]),
                    int(item["dst_core"]),
                    int(item["size_bytes"]),
                    str(item["payload"]),
                )
            )
        env.run()
        return env.now

    makespan_k8n2 = run_collective(8, 2)
    makespan_k2n6 = run_collective(2, 6)
    assert makespan_k8n2 > 0
    assert makespan_k2n6 <= makespan_k8n2 * 2

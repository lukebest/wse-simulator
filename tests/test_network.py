from __future__ import annotations

import simpy

from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.routing.table_based import TableBasedRouting
from wsesim.network.topology.mesh2d import Mesh2D


def test_network_packet_delivery_mesh() -> None:
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=Mesh2D(),
        routing=DimensionOrderRouting(),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=16,
        link_bw_flits_per_cycle=2,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
    )
    env.process(net.send_packet(Packet(src=0, dst=15, size_bytes=128, payload_type="weight")))
    env.run()
    assert net.stats.packets_sent == 1
    assert net.stats.max_packet_latency > 0


def test_table_based_routing_with_removed_link() -> None:
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=Mesh2D(),
        routing=DimensionOrderRouting(),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=16,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
    )
    dead = {(0, 1), (1, 0)}
    net.remove_dead_components(dead_links=dead)
    net.routing = TableBasedRouting(net.graph)
    env.process(net.send_packet(Packet(src=0, dst=15, size_bytes=64, payload_type="activation")))
    env.run()
    assert net.stats.packets_sent == 1


def test_router_4stage_pipeline_increases_latency_vs_1stage() -> None:
    def run(mode: str) -> float:
        env = simpy.Environment()
        net = UnifiedNetwork(
            env=env,
            topology=Mesh2D(),
            routing=DimensionOrderRouting(),
            flow_control=CreditBasedVCFlowControl(),
            num_nodes=16,
            link_bw_flits_per_cycle=2,
            link_latency_cycles=1,
            num_vcs=2,
            buffer_depth=8,
            router_pipeline_mode=mode,
            rc_latency_cycles=1,
            va_latency_cycles=1,
            sa_latency_cycles=1,
            st_latency_cycles=1,
            crossbar_bw_flits_per_cycle=2,
        )
        env.process(
            net.send_packet(Packet(src=0, dst=15, size_bytes=128, payload_type="activation"))
        )
        env.run()
        return net.stats.avg_latency()

    latency_1stage = run("1_stage")
    latency_4stage = run("4_stage")
    assert latency_4stage > latency_1stage


def test_router_pipeline_drains_input_buffer_under_concurrency() -> None:
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=Mesh2D(),
        routing=DimensionOrderRouting(),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=16,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
        router_pipeline_mode="4_stage",
        rc_latency_cycles=1,
        va_latency_cycles=1,
        sa_latency_cycles=1,
        st_latency_cycles=1,
        crossbar_bw_flits_per_cycle=1,
    )

    # Send several packets through overlapping routes to stress router buffering.
    packets = [
        Packet(src=0, dst=15, size_bytes=128, payload_type="activation"),
        Packet(src=1, dst=14, size_bytes=128, payload_type="activation"),
        Packet(src=2, dst=13, size_bytes=128, payload_type="activation"),
        Packet(src=3, dst=12, size_bytes=128, payload_type="activation"),
    ]
    for pkt in packets:
        env.process(net.send_packet(pkt))
    env.run()

    assert net.stats.packets_sent == len(packets)
    # If dequeue is modeled correctly, steady-state queues should eventually drain.
    assert all(len(router.input_buffer.items) == 0 for router in net.routers.values())

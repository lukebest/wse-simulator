from __future__ import annotations

import simpy

from wsesim.core.stats import SimResult
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


def test_network_packet_delivery_rect_mesh() -> None:
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=Mesh2D(rows=6, cols=8),
        routing=DimensionOrderRouting(),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=48,
        link_bw_flits_per_cycle=2,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
    )
    env.process(net.send_packet(Packet(src=0, dst=47, size_bytes=128, payload_type="weight")))
    env.run()
    assert net.stats.packets_sent == 1
    assert net.stats.max_packet_latency > 0


def test_network_packet_delivery_from_io_node_rect_mesh() -> None:
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=Mesh2D(rows=6, cols=8),
        routing=DimensionOrderRouting(),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=48,
        link_bw_flits_per_cycle=2,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
    )
    # IO node at (row=1, col=0) maps to physical node 8 in 8-column mesh.
    env.process(net.send_packet(Packet(src=8, dst=47, size_bytes=128, payload_type="token_dispatch")))
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


def test_flit_level_model_large_packet_has_higher_latency() -> None:
    def run(size_bytes: int) -> tuple[float, int]:
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
        env.process(
            net.send_packet(Packet(src=0, dst=15, size_bytes=size_bytes, payload_type="weight"))
        )
        env.run()
        return net.stats.avg_latency(), net.stats.flits_sent

    small_latency, small_flits = run(32)
    large_latency, large_flits = run(256)
    assert large_flits > small_flits
    assert large_latency > small_latency


def test_vc_is_reserved_per_packet_until_tail() -> None:
    env = simpy.Environment()
    net = UnifiedNetwork(
        env=env,
        topology=Mesh2D(),
        routing=DimensionOrderRouting(),
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=16,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=1,  # force contention on VC reservation
        buffer_depth=8,
        router_pipeline_mode="4_stage",
        rc_latency_cycles=1,
        va_latency_cycles=1,
        sa_latency_cycles=1,
        st_latency_cycles=1,
        crossbar_bw_flits_per_cycle=1,
    )
    pkt_a = Packet(src=0, dst=15, size_bytes=256, payload_type="activation")
    pkt_b = Packet(src=0, dst=15, size_bytes=256, payload_type="activation")
    env.process(net.send_packet(pkt_a))
    env.process(net.send_packet(pkt_b))
    env.run()

    assert net.stats.packets_sent == 2
    # Both packets share the same path and single VC; second packet should wait.
    assert net.stats.avg_latency() > 0
    assert net.stats.vc_wait_cycles > 0
    assert net.stats.pipeline_cycles > 0
    assert len(net.stats.per_router_vc_wait_cycles) > 0
    assert all(len(router.active_vc_packets) == 0 for router in net.routers.values())

    sim_result = SimResult(total_latency_cycles=int(env.now))
    sim_result.update_from_network_stats(net.stats, sim_time_cycles=int(env.now))
    assert sim_result.network_cycles > 0
    assert sim_result.network_throughput > 0


def test_ring_allreduce_sequential_generates_correct_traffic() -> None:
    from wsesim.network.collective import generate_ring_allreduce_traffic

    nodes = [0, 1, 2, 3]
    traffic = generate_ring_allreduce_traffic(
        participating_nodes=nodes,
        payload_bytes_per_expert=1024,
        num_experts=2,
        strategy="sequential",
    )
    assert len(traffic) > 0
    for item in traffic:
        assert item["src_core"] in nodes
        assert item["dst_core"] in nodes
        assert item["src_core"] != item["dst_core"]
        assert item["size_bytes"] > 0
        assert item["payload"] in ("allreduce_rs", "allreduce_ag")
    S = len(nodes)
    expected_items_per_expert = 2 * (S - 1) * S
    assert len(traffic) == expected_items_per_expert * 2


def test_ring_allreduce_entwined_generates_staggered_traffic() -> None:
    from wsesim.network.collective import generate_ring_allreduce_traffic

    nodes = [0, 1, 2, 3]
    traffic = generate_ring_allreduce_traffic(
        participating_nodes=nodes,
        payload_bytes_per_expert=1024,
        num_experts=3,
        strategy="entwined",
    )
    assert len(traffic) > 0
    delays = {item["delay_cycles"] for item in traffic}
    assert len(delays) > 1, "Entwined mode should produce varied delay offsets"


def test_ring_allreduce_empty_cases() -> None:
    from wsesim.network.collective import generate_ring_allreduce_traffic

    assert generate_ring_allreduce_traffic([0], 1024, 2) == []
    assert generate_ring_allreduce_traffic([0, 1], 1024, 0) == []
    assert generate_ring_allreduce_traffic([0, 1], 0, 2) == []

"""Ordering tests for per-color FIFO guarantees."""

from __future__ import annotations

import simpy

from wsesim.network.color import ColorPlan
from wsesim.network.color_network import ColorNetwork
from wsesim.network.color_routes import build_path_color
from wsesim.network.ordering import OrderTracker
from wsesim.network.packet import Packet
from wsesim.network.topology.mesh2d import Mesh2D


def test_sequential_packets_same_color_preserve_order() -> None:
    plan = ColorPlan.empty(16, 4)
    build_path_color(plan, 0, 0, 15, 4, 4)
    env = simpy.Environment()
    net = ColorNetwork(
        env=env,
        topology=Mesh2D(),
        color_plan=plan,
        num_nodes=16,
        enforce_single_source=True,
    )

    def send_all():
        for seq in range(3):
            yield env.process(
                net.send_packet(
                    Packet(
                        src=0,
                        dst=15,
                        size_bytes=64,
                        payload_type="test",
                        color=0,
                        seq=seq,
                    )
                )
            )

    env.process(send_all())
    env.run()
    net.finalize_stats()
    assert net.order_tracker.violations == 0


def test_order_tracker_fifo() -> None:
    ot = OrderTracker()
    for i in range(5):
        ot.record_delivery(0, 1, i, float(i))
    assert ot.violations == 0

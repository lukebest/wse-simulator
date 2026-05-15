"""Run a single packet transfer simulation."""

from __future__ import annotations

import simpy

from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.topology.mesh2d import Mesh2D


def main() -> None:
    env = simpy.Environment()
    network = UnifiedNetwork(
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
    pkt = Packet(src=0, dst=15, size_bytes=256, payload_type="activation")
    env.process(network.send_packet(pkt))
    env.run()
    print("packets_sent:", network.stats.packets_sent)
    print("avg_latency:", network.stats.avg_latency())


if __name__ == "__main__":
    main()

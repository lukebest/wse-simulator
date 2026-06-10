from __future__ import annotations

import simpy

from wsesim.network.collective import generate_collective_traffic
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.topology.mesh2d import Mesh2D
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly


def _run_allgather(topology_key: str, algorithm: str, slot_cycles: int = 1) -> UnifiedNetwork:
    env = simpy.Environment()
    if topology_key == "mesh2d_4x4_ps":
        topology = Mesh2D(rows=4, cols=4)
        routing = DimensionOrderRouting()
        hint = {"rows": 4, "cols": 4}
    else:
        _, _, k_token, n_token = topology_key.split("_")
        k = int(k_token[1:])
        n = int(n_token[1:])
        topology = TDMFlatButterfly(k=k, n=n, rows=4, cols=4)
        routing = TDMFlatButterflyRouting(topology=topology)
        hint = {"k": k, "n": n}
    net = UnifiedNetwork(
        env=env,
        topology=topology,
        routing=routing,
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=16,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
        slot_cycles=slot_cycles,
    )
    traffic = generate_collective_traffic(
        algorithm=algorithm,
        participating_nodes_global=list(range(16)),
        cores_per_reticle=16,
        payload_bytes_per_expert=1024,
        num_experts=1,
        topology_hint=hint,
    )
    for item in traffic:
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
    return net


def test_allgather_4x4_end_to_end_mesh_and_tdm() -> None:
    cases = [
        ("mesh2d_4x4_ps", "direct_allgather", 1),
        ("tdm_fb_k4_n2", "direct_allgather", 1),
        ("tdm_fb_k2_n4", "nd_dimension_exchange_allgather", 4),
    ]
    for topology_key, algorithm, slot_cycles in cases:
        net = _run_allgather(topology_key, algorithm, slot_cycles)
        assert net.stats.packets_sent > 0
        assert net.stats.flits_sent > 0
        assert net.stats.max_packet_latency > 0


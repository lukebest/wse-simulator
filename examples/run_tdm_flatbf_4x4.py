"""Run 4x4 TDM flattened-butterfly collective comparisons."""

from __future__ import annotations

import csv
from pathlib import Path

import simpy

from wsesim.network.collective import generate_collective_traffic
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Packet
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.topology.mesh2d import Mesh2D
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly


NODE_COUNT = 16
ROWS = 4
COLS = 4


def _build_network(topology_key: str) -> UnifiedNetwork:
    env = simpy.Environment()
    if topology_key == "mesh2d_4x4_ps":
        topology = Mesh2D(rows=ROWS, cols=COLS)
        routing = DimensionOrderRouting()
    else:
        _, _, k_token, n_token = topology_key.split("_")
        k = int(k_token[1:])
        n = int(n_token[1:])
        topology = TDMFlatButterfly(k=k, n=n, rows=ROWS, cols=COLS)
        routing = TDMFlatButterflyRouting(topology=topology)
    return UnifiedNetwork(
        env=env,
        topology=topology,
        routing=routing,
        flow_control=CreditBasedVCFlowControl(),
        num_nodes=NODE_COUNT,
        link_bw_flits_per_cycle=1,
        link_latency_cycles=1,
        num_vcs=2,
        buffer_depth=8,
    )


def _collective_algo(topology_key: str, collective: str) -> str:
    if topology_key == "mesh2d_4x4_ps":
        return "2d_ring" if collective == "allreduce" else "direct_allgather"
    return "nd_dimension_exchange_allreduce" if collective == "allreduce" else "nd_dimension_exchange_allgather"


def _topology_hint(topology_key: str) -> dict[str, int]:
    if topology_key == "mesh2d_4x4_ps":
        return {"rows": ROWS, "cols": COLS}
    _, _, k_token, n_token = topology_key.split("_")
    return {"k": int(k_token[1:]), "n": int(n_token[1:])}


def _run_case(
    topology_key: str, collective: str, msg_bytes: int, simulation_scale: int
) -> dict[str, float | int | str]:
    net = _build_network(topology_key)
    env = net.env
    algorithm = _collective_algo(topology_key, collective)
    traffic = generate_collective_traffic(
        algorithm=algorithm,
        participating_nodes_global=list(range(NODE_COUNT)),
        cores_per_reticle=NODE_COUNT,
        payload_bytes_per_expert=max(32, msg_bytes // max(1, simulation_scale)),
        num_experts=1,
        topology_hint=_topology_hint(topology_key),
    )
    for item in traffic:
        delay = int(item.get("delay_cycles", 0))

        def _inject(
            sim_env: simpy.Environment,
            delay_cycles: int,
            src: int,
            dst: int,
            size_bytes: int,
            payload: str,
        ):
            if delay_cycles > 0:
                yield sim_env.timeout(delay_cycles)
            yield sim_env.process(
                net.send_packet(Packet(src=src, dst=dst, size_bytes=size_bytes, payload_type=payload))
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
    sim_time = max(1.0, float(env.now))
    avg_link_util = (
        sum(link.total_busy_cycles for link in net.links.values()) / (len(net.links) * sim_time)
        if net.links
        else 0.0
    )
    return {
        "topology": topology_key,
        "collective": collective,
        "msg_bytes": msg_bytes,
        "simulated_msg_bytes": max(32, msg_bytes // max(1, simulation_scale)),
        "makespan_cycles": int(env.now),
        "avg_latency": float(net.stats.avg_latency()),
        "avg_link_util": float(avg_link_util),
        "total_flits": int(net.stats.flits_sent),
    }


def main() -> None:
    simulation_scale = 16
    topologies = ["mesh2d_4x4_ps", "tdm_fb_k4_n2", "tdm_fb_k2_n4"]
    collectives = ["allreduce", "allgather"]
    message_sizes = [1024, 16 * 1024, 256 * 1024]

    rows: list[dict[str, float | int | str]] = []
    for topo in topologies:
        for collective in collectives:
            for msg_bytes in message_sizes:
                rows.append(_run_case(topo, collective, msg_bytes, simulation_scale))

    out_dir = Path("outputs/tdm_flatbf_4x4")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topology",
                "collective",
                "msg_bytes",
                "simulated_msg_bytes",
                "makespan_cycles",
                "avg_latency",
                "avg_link_util",
                "total_flits",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

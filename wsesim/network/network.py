"""Unified network model for NoC/NoW."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import simpy

from wsesim.network.flow_control.base import FlowControl
from wsesim.network.link import Link
from wsesim.network.packet import Packet, packet_to_flits
from wsesim.network.router import Router
from wsesim.network.routing.base import RoutingAlgorithm
from wsesim.network.topology.base import Topology


@dataclass(slots=True)
class NetworkStats:
    packets_sent: int = 0
    flits_sent: int = 0
    total_packet_latency: float = 0.0
    max_packet_latency: float = 0.0
    total_hops: int = 0
    vc_wait_cycles: int = 0
    buffer_wait_cycles: int = 0
    pipeline_cycles: int = 0
    link_wait_cycles: int = 0
    per_router_vc_wait_cycles: dict[int, int] = field(default_factory=dict)
    per_router_buffer_wait_cycles: dict[int, int] = field(default_factory=dict)
    per_router_pipeline_cycles: dict[int, int] = field(default_factory=dict)

    def avg_latency(self) -> float:
        return 0.0 if self.packets_sent == 0 else self.total_packet_latency / self.packets_sent


@dataclass(slots=True)
class UnifiedNetwork:
    env: simpy.Environment
    topology: Topology
    routing: RoutingAlgorithm
    flow_control: FlowControl
    num_nodes: int
    link_bw_flits_per_cycle: int
    link_latency_cycles: int
    num_vcs: int
    buffer_depth: int
    router_pipeline_mode: str = "4_stage"
    rc_latency_cycles: int = 1
    va_latency_cycles: int = 1
    sa_latency_cycles: int = 1
    st_latency_cycles: int = 1
    crossbar_bw_flits_per_cycle: int = 1
    flit_bytes: int = 128
    graph: dict[int, list[int]] = field(init=False)
    routers: dict[int, Router] = field(init=False)
    links: dict[tuple[int, int], Link] = field(init=False)
    stats: NetworkStats = field(default_factory=NetworkStats)

    def __post_init__(self) -> None:
        self.graph = self.topology.build(self.num_nodes)
        self.routers = {
            node: Router(
                env=self.env,
                node_id=node,
                num_vcs=self.num_vcs,
                buffer_depth=self.buffer_depth,
                pipeline_mode=self.router_pipeline_mode,
                routing_latency_cycles=self.rc_latency_cycles,
                vc_alloc_latency_cycles=self.va_latency_cycles,
                switch_alloc_latency_cycles=self.sa_latency_cycles,
                switch_traversal_latency_cycles=self.st_latency_cycles,
                crossbar_bw_flits_per_cycle=self.crossbar_bw_flits_per_cycle,
            )
            for node in self.graph
        }
        self.links = {}
        for src, dsts in self.graph.items():
            for dst in dsts:
                self.links[(src, dst)] = Link(
                    env=self.env,
                    src=src,
                    dst=dst,
                    bandwidth_flits_per_cycle=self.link_bw_flits_per_cycle,
                    latency_cycles=self.link_latency_cycles,
                )

    def remove_dead_components(
        self, dead_nodes: set[int] | None = None, dead_links: set[tuple[int, int]] | None = None
    ) -> None:
        dead_nodes = dead_nodes or set()
        dead_links = dead_links or set()

        for dead in dead_nodes:
            self.graph.pop(dead, None)
            self.routers.pop(dead, None)

        for src in list(self.graph):
            self.graph[src] = [
                dst
                for dst in self.graph[src]
                if dst not in dead_nodes and (src, dst) not in dead_links
            ]
        self.links = {
            (src, dst): link
            for (src, dst), link in self.links.items()
            if src not in dead_nodes and dst not in dead_nodes and (src, dst) not in dead_links
        }

    def send_packet(self, packet: Packet):
        if packet.src not in self.graph or packet.dst not in self.graph:
            raise ValueError("Packet source/destination is unavailable in graph.")

        start = self.env.now
        packet_id = id(packet)
        hops = 0
        current = packet.src
        flits = packet_to_flits(packet, flit_bytes=self.flit_bytes)

        while current != packet.dst:
            router = self.routers[current]
            next_hop = self.routing.next_hop(current, packet.dst, self.graph)
            if next_hop not in self.graph.get(current, []):
                raise ValueError(f"Invalid next hop {next_hop} from {current}.")
            next_router = self.routers[next_hop]

            for flit in flits:
                if flit.is_head:
                    while not next_router.can_reserve_vc(packet_id):
                        self.stats.vc_wait_cycles += 1
                        router.vc_wait_cycles += 1
                        self.stats.per_router_vc_wait_cycles[router.node_id] = (
                            self.stats.per_router_vc_wait_cycles.get(router.node_id, 0) + 1
                        )
                        yield self.env.timeout(1)
                    if not next_router.reserve_vc(packet_id):
                        self.stats.vc_wait_cycles += 1
                        router.vc_wait_cycles += 1
                        self.stats.per_router_vc_wait_cycles[router.node_id] = (
                            self.stats.per_router_vc_wait_cycles.get(router.node_id, 0) + 1
                        )
                        yield self.env.timeout(1)
                        continue

                while not self.flow_control.can_send(
                    len(next_router.input_buffer.items), next_router.input_buffer.capacity
                ):
                    self.stats.buffer_wait_cycles += 1
                    router.buffer_wait_cycles += 1
                    self.stats.per_router_buffer_wait_cycles[router.node_id] = (
                        self.stats.per_router_buffer_wait_cycles.get(router.node_id, 0) + 1
                    )
                    yield self.env.timeout(1)
                while len(router.input_buffer.items) >= router.input_buffer.capacity:
                    self.stats.buffer_wait_cycles += 1
                    router.buffer_wait_cycles += 1
                    self.stats.per_router_buffer_wait_cycles[router.node_id] = (
                        self.stats.per_router_buffer_wait_cycles.get(router.node_id, 0) + 1
                    )
                    yield self.env.timeout(1)
                yield router.enqueue(flit)
                pipeline_before = router.pipeline_cycles
                yield self.env.process(router.pipeline(1))
                self.stats.pipeline_cycles += router.pipeline_cycles - pipeline_before
                self.stats.per_router_pipeline_cycles[router.node_id] = (
                    self.stats.per_router_pipeline_cycles.get(router.node_id, 0)
                    + (router.pipeline_cycles - pipeline_before)
                )

                link = self.links[(current, next_hop)]
                wait_before = link.total_wait_cycles
                yield self.env.process(link.transfer(1))
                self.stats.link_wait_cycles += link.total_wait_cycles - wait_before
                self.stats.flits_sent += 1

                if flit.is_tail:
                    next_router.release_vc(packet_id)
            hops += 1
            current = next_hop

        latency = self.env.now - start
        self.stats.packets_sent += 1
        self.stats.total_packet_latency += latency
        self.stats.max_packet_latency = max(self.stats.max_packet_latency, latency)
        self.stats.total_hops += hops

    def estimate_transfer_cycles(self, size_bytes: int) -> int:
        flits = max(1, ceil(size_bytes / max(self.flit_bytes, 1)))
        crossbar_cycles = ceil(flits / max(self.crossbar_bw_flits_per_cycle, 1))
        if self.router_pipeline_mode == "1_stage":
            router_cycles = max(
                self.rc_latency_cycles,
                self.va_latency_cycles,
                self.sa_latency_cycles,
                self.st_latency_cycles,
            ) + crossbar_cycles
        else:
            router_cycles = (
                self.rc_latency_cycles
                + self.va_latency_cycles
                + self.sa_latency_cycles
                + self.st_latency_cycles
                + crossbar_cycles
            )
        link_cycles = self.link_latency_cycles + ceil(flits / max(self.link_bw_flits_per_cycle, 1))
        return router_cycles + link_cycles

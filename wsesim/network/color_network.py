"""Color virtual-network simulation on physical 2D mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import simpy

from wsesim.network.color import ColorPlan
from wsesim.network.edge_routes import EdgeRouteTable
from wsesim.network.color_router import ColorRouter
from wsesim.network.flow_control.per_color_credit import PerColorCreditFlowControl
from wsesim.network.link import Link
from wsesim.network.ordering import OrderTracker
from wsesim.network.packet import Flit, Packet, packet_to_flits
from wsesim.network.topology.base import Topology


@dataclass(slots=True)
class ColorNetworkStats:
    packets_sent: int = 0
    flits_sent: int = 0
    total_packet_latency: float = 0.0
    max_packet_latency: float = 0.0
    total_hops: int = 0
    color_buffer_wait_cycles: int = 0
    pipeline_cycles: int = 0
    link_wait_cycles: int = 0
    ordering_violations: int = 0
    per_router_color_wait: dict[int, int] = field(default_factory=dict)

    def avg_latency(self) -> float:
        return 0.0 if self.packets_sent == 0 else self.total_packet_latency / self.packets_sent


@dataclass(slots=True)
class ColorNetwork:
    """Patent-faithful color NoC on physical mesh."""

    env: simpy.Environment
    topology: Topology
    color_plan: ColorPlan
    num_nodes: int
    edge_routes: EdgeRouteTable | None = None
    link_bw_flits_per_cycle: int = 1
    link_latency_cycles: int = 1
    entries_per_color: int = 2
    pipeline_cycles: int = 1
    flit_bytes: int = 128
    enforce_single_source: bool = True
    active_color_sources: dict[int, int | None] = field(default_factory=dict)
    graph: dict[int, list[int]] = field(init=False)
    routers: dict[int, ColorRouter] = field(init=False)
    links: dict[tuple[int, int], Link] = field(init=False)
    flow_control: PerColorCreditFlowControl = field(init=False)
    stats: ColorNetworkStats = field(default_factory=ColorNetworkStats)
    order_tracker: OrderTracker = field(default_factory=OrderTracker)
    trace: list[dict] = field(default_factory=list)
    enable_trace: bool = False
    watchdog_cycles: int = 1_000_000
    deadlock_detected: bool = False
    color_occupancy: dict[int, dict[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.graph = self.topology.build(self.num_nodes)
        self.flow_control = PerColorCreditFlowControl(entries_per_color=self.entries_per_color)
        nc = self.color_plan.num_colors
        self.routers = {
            node: ColorRouter(
                env=self.env,
                node_id=node,
                num_colors=nc,
                entries_per_color=self.entries_per_color,
                pipeline_cycles=self.pipeline_cycles,
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
        for c in range(nc):
            self.active_color_sources[c] = None
        self.color_occupancy = {node: {c: 0 for c in range(nc)} for node in self.graph}

    @property
    def num_colors(self) -> int:
        return self.color_plan.num_colors

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
                d for d in self.graph[src] if d not in dead_nodes and (src, d) not in dead_links
            ]
        self.links = {
            (s, d): lk
            for (s, d), lk in self.links.items()
            if s not in dead_nodes and d not in dead_nodes and (s, d) not in dead_links
        }

    def _acquire_color_source(self, color: int, src: int) -> None:
        if not self.enforce_single_source:
            return
        active = self.active_color_sources.get(color)
        if active is None:
            self.active_color_sources[color] = src
        elif active != src:
            raise RuntimeError(f"Color {color} already active from {active}, cannot start {src}")

    def _release_color_source(self, color: int, src: int) -> None:
        if not self.enforce_single_source:
            return
        if self.active_color_sources.get(color) == src:
            self.active_color_sources[color] = None

    def _trace_event(self, **kwargs) -> None:
        if self.enable_trace:
            kwargs["cycle"] = int(self.env.now)
            self.trace.append(kwargs)

    def send_packet(self, packet: Packet):
        """Route packet along fixed color path from src to dst."""
        if packet.src not in self.graph or packet.dst not in self.graph:
            raise ValueError("Packet source/destination unavailable.")
        color = packet.color
        self._acquire_color_source(color, packet.src)
        start = self.env.now
        flits = packet_to_flits(packet, flit_bytes=self.flit_bytes)
        hops = 0
        try:
            yield from self._forward_flits(packet, flits, hops)
        finally:
            if flits and flits[-1].is_tail:
                self._release_color_source(color, packet.src)

        latency = self.env.now - start
        self.stats.packets_sent += 1
        self.stats.total_packet_latency += latency
        self.stats.max_packet_latency = max(self.stats.max_packet_latency, latency)

    def _forward_flits(self, packet: Packet, flits: list[Flit], hops: int):
        """Forward all flits along precomputed path or color route."""
        color = packet.color
        if self.edge_routes is not None:
            path = self.edge_routes.path_for(packet.src, packet.dst)
        else:
            path = self.color_plan.trace_route(packet.src, packet.dst, color)
        if path is None:
            raise ValueError(f"No route from {packet.src} to {packet.dst} (color {color})")

        for step_idx in range(len(path) - 1):
            current, nxt = path[step_idx], path[step_idx + 1]
            if (current, nxt) not in self.links:
                raise ValueError(f"Dead link {current}->{nxt} on color path")
            router = self.routers[current]
            for flit in flits:
                while not self.flow_control.can_send(
                    self.color_occupancy[nxt][color],
                    self.entries_per_color,
                ):
                    self.stats.color_buffer_wait_cycles += 1
                    self.stats.per_router_color_wait[current] = (
                        self.stats.per_router_color_wait.get(current, 0) + 1
                    )
                    router.color_buffer_wait_cycles += 1
                    self._trace_event(
                        stage="color_wait",
                        node=current,
                        color=color,
                        next_hop=nxt,
                        pkt_src=packet.src,
                        pkt_dst=packet.dst,
                    )
                    yield self.env.timeout(1)

                self.color_occupancy[nxt][color] += 1
                yield self.env.process(router.pipeline(1))
                self.stats.pipeline_cycles += router.pipeline_cycles

                link = self.links[(current, nxt)]
                wait_before = link.total_wait_cycles
                yield self.env.process(link.transfer(1))
                self.stats.link_wait_cycles += link.total_wait_cycles - wait_before
                self.stats.flits_sent += 1
                # Flit consumed at downstream router after link traversal
                self.color_occupancy[nxt][color] = max(0, self.color_occupancy[nxt][color] - 1)
                self._trace_event(
                    stage="link_tx",
                    node=current,
                    next_hop=nxt,
                    link=f"{current}->{nxt}",
                    color=color,
                    pkt_src=packet.src,
                    pkt_dst=packet.dst,
                )

            hops += 1

        self.stats.total_hops += hops
        if packet.dst == path[-1]:
            if self.enforce_single_source:
                self.order_tracker.record_delivery(packet.src, color, packet.seq, self.env.now)
            self._trace_event(
                stage="done",
                node=packet.dst,
                color=color,
                pkt_src=packet.src,
                pkt_dst=packet.dst,
            )

    def send_multicast_packet(self, packet: Packet, targets: list[int] | None = None):
        """Send along multicast color tree (replicate at branch points)."""
        color = packet.color
        self._acquire_color_source(color, packet.src)
        flits = packet_to_flits(packet, flit_bytes=self.flit_bytes)
        visited: set[int] = set()
        queue: list[tuple[int, list[int]]] = [(packet.src, flits)]
        start = self.env.now
        try:
            while queue:
                current, current_flits = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                hops = self.color_plan.next_hops(current, color)
                if not hops:
                    if current == packet.dst or (targets and current in targets):
                        self.order_tracker.record_delivery(
                            packet.src, color, packet.seq, self.env.now
                        )
                    continue
                for nxt in hops:
                    if (current, nxt) not in self.links:
                        continue
                    router = self.routers[current]
                    next_router = self.routers[nxt]
                    branch_flits = [
                        Flit(
                            packet=flit.packet,
                            flit_id=flit.flit_id,
                            is_head=flit.is_head,
                            is_tail=flit.is_tail,
                            color=color,
                            branch_id=hash(nxt) % 1000,
                        )
                        for flit in current_flits
                    ]
                    for flit in branch_flits:
                        while not self.flow_control.can_send(
                            self.color_occupancy[nxt][color],
                            self.entries_per_color,
                        ):
                            self.stats.color_buffer_wait_cycles += 1
                            yield self.env.timeout(1)
                        self.color_occupancy[nxt][color] += 1
                        yield router.enqueue(color, flit)
                        yield self.env.process(router.pipeline(1))
                        link = self.links[(current, nxt)]
                        yield self.env.process(link.transfer(1))
                        self.stats.flits_sent += 1
                        self.color_occupancy[nxt][color] = max(
                            0, self.color_occupancy[nxt][color] - 1
                        )
                    queue.append((nxt, branch_flits))
        finally:
            self._release_color_source(color, packet.src)

        latency = self.env.now - start
        self.stats.packets_sent += 1
        self.stats.total_packet_latency += latency

    def avg_link_utilization(self) -> float:
        if not self.links:
            return 0.0
        total_busy = sum(lk.total_busy_cycles for lk in self.links.values())
        sim_time = max(1, int(self.env.now))
        return total_busy / (len(self.links) * sim_time)

    def finalize_stats(self) -> None:
        self.stats.ordering_violations = self.order_tracker.violations

    def estimate_transfer_cycles(self, size_bytes: int, hops: int = 1) -> int:
        flits = max(1, ceil(size_bytes / max(self.flit_bytes, 1)))
        link_cycles = self.link_latency_cycles + ceil(flits / max(self.link_bw_flits_per_cycle, 1))
        return hops * (self.pipeline_cycles + link_cycles)

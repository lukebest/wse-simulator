"""Color-aware NoC simulation on physical 2D mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import simpy

from wsesim.network.color import ColorPlan
from wsesim.network.color_router import ColorRouter
from wsesim.network.flow_control.per_color_credit import PerColorCreditFlowControl
from wsesim.network.link import Link
from wsesim.network.packet import Flit, Packet, packet_to_flits
from wsesim.network.topology.mesh2d import Mesh2D


@dataclass(slots=True)
class ColorNetworkStats:
    packets_sent: int = 0
    flits_sent: int = 0
    total_packet_latency: float = 0.0
    max_packet_latency: float = 0.0
    total_hops: int = 0
    color_buffer_wait_cycles: int = 0
    buffer_wait_cycles: int = 0
    pipeline_cycles: int = 0
    link_wait_cycles: int = 0
    ordering_violations: int = 0

    def avg_latency(self) -> float:
        return 0.0 if self.packets_sent == 0 else self.total_packet_latency / self.packets_sent


@dataclass(slots=True)
class ColorNetwork:
    env: simpy.Environment
    plan: ColorPlan
    link_bw_flits_per_cycle: int = 1
    link_latency_cycles: int = 1
    entries_per_color: int = 2
    flit_bytes: int = 128
    router_pipeline_mode: str = "1_stage"
    graph: dict[int, list[int]] = field(init=False)
    routers: dict[int, ColorRouter] = field(init=False)
    links: dict[tuple[int, int], Link] = field(init=False)
    flow_control: PerColorCreditFlowControl = field(init=False)
    stats: ColorNetworkStats = field(default_factory=ColorNetworkStats)
    delivery_seq: dict[tuple[int, int, int], list[int]] = field(default_factory=dict)
    seq_counter: dict[tuple[int, int, int], int] = field(default_factory=dict)
    stream_locks: dict[tuple[int, int, int], simpy.Resource] = field(default_factory=dict)
    watchdog_limit: int = 1_000_000

    def __post_init__(self) -> None:
        n = self.plan.num_nodes
        self.graph = Mesh2D(rows=self.plan.rows, cols=self.plan.cols).build(n)
        self.flow_control = PerColorCreditFlowControl(self.entries_per_color)
        self.routers = {
            node: ColorRouter(
                env=self.env,
                node_id=node,
                plan=self.plan,
                entries_per_color=self.entries_per_color,
                pipeline_mode=self.router_pipeline_mode,
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

    def _stream_lock(self, color: int, src: int, dst: int) -> simpy.Resource:
        key = (color, src, dst)
        if key not in self.stream_locks:
            self.stream_locks[key] = simpy.Resource(self.env, capacity=1)
        return self.stream_locks[key]

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
            k: v
            for k, v in self.links.items()
            if k[0] not in dead_nodes and k[1] not in dead_nodes and k not in dead_links
        }

    def _record_delivery(self, color: int, src: int, dst: int, seq: int) -> None:
        key = (color, src, dst)
        seqs = self.delivery_seq.setdefault(key, [])
        if seqs and seq < seqs[-1]:
            self.stats.ordering_violations += 1
        seqs.append(seq)

    def send_packet(self, packet: Packet, color: int = 0, seq: int | None = None):
        if packet.src not in self.graph or packet.dst not in self.graph:
            raise ValueError("Packet source/destination unavailable.")

        lock = self._stream_lock(color, packet.src, packet.dst)
        req = lock.request()
        yield req

        key = (color, packet.src, packet.dst)
        if seq is None:
            seq = self.seq_counter.get(key, 0)
            self.seq_counter[key] = seq + 1

        start = self.env.now
        flits = packet_to_flits(packet, flit_bytes=self.flit_bytes)
        hops = 0
        current = packet.src

        try:
            for flit in flits:
                while current != packet.dst:
                    router = self.routers[current]
                    dst_router = None
                    next_hop = self._select_next_hop(current, packet.dst, color)
                    if next_hop is None:
                        raise ValueError(
                            f"No route at node {current} for dst {packet.dst} color {color}"
                        )
                    dst_router = self.routers[next_hop]

                    while router.is_stalled(color):
                        self.stats.color_buffer_wait_cycles += 1
                        router.color_buffer_wait_cycles += 1
                        yield self.env.timeout(1)
                    while not self.flow_control.can_send(
                        dst_router.queue_len(color), self.entries_per_color
                    ):
                        self.stats.color_buffer_wait_cycles += 1
                        self.stats.buffer_wait_cycles += 1
                        router.stall_out[color] = True
                        yield self.env.timeout(1)
                    router.stall_out[color] = False

                    pipeline_before = router.pipeline_cycles
                    yield self.env.process(router.pipeline(1))
                    self.stats.pipeline_cycles += router.pipeline_cycles - pipeline_before

                    link = self.links[(current, next_hop)]
                    wait_before = link.total_wait_cycles
                    yield self.env.process(link.transfer(1))
                    self.stats.link_wait_cycles += link.total_wait_cycles - wait_before
                    self.stats.flits_sent += 1
                    dst_router.update_stall(color)

                    current = next_hop
                    hops += 1

                if flit.is_tail:
                    self._record_delivery(color, packet.src, packet.dst, seq)

            latency = self.env.now - start
            self.stats.packets_sent += 1
            self.stats.total_packet_latency += latency
            self.stats.max_packet_latency = max(self.stats.max_packet_latency, latency)
            self.stats.total_hops += hops
        finally:
            lock.release(req)

    def _select_next_hop(self, current: int, dst: int, color: int) -> int | None:
        hops = self.plan.next_hops(current, color, dst)
        valid = [h for h in hops if h in self.graph.get(current, [])]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        # Multicast replication handled by caller; unicast picks neighbor closest to dst.
        return min(valid, key=lambda h: _mesh_manhattan(h, dst, self.plan.cols))

    def _transfer_flit(self, flit: Flit, color: int, src: int, dst: int):
        del flit
        if (src, dst) not in self.links:
            raise ValueError(f"No link {src}->{dst}")
        dst_router = self.routers[dst]
        while not self.flow_control.can_send(dst_router.queue_len(color), self.entries_per_color):
            self.stats.color_buffer_wait_cycles += 1
            self.stats.buffer_wait_cycles += 1
            self.routers[src].stall_out[color] = True
            yield self.env.timeout(1)
        self.routers[src].stall_out[color] = False
        link = self.links[(src, dst)]
        wait_before = link.total_wait_cycles
        yield self.env.process(link.transfer(1))
        self.stats.link_wait_cycles += link.total_wait_cycles - wait_before
        self.stats.flits_sent += 1
        dst_router.update_stall(color)

    def avg_link_util(self) -> float:
        if not self.links:
            return 0.0
        total_busy = sum(l.total_busy_cycles for l in self.links.values())
        horizon = max(1, int(self.env.now))
        return total_busy / (len(self.links) * horizon)

    def run_traffic(self, traffic: list[dict], color_for_payload: dict[str, int] | None = None):
        color_map = color_for_payload or {}
        procs = []
        for idx, pkt in enumerate(traffic):
            delay = int(pkt.get("delay_cycles", 0))
            color = int(pkt.get("color", color_map.get(pkt.get("payload", ""), 0)))
            p = Packet(
                src=int(pkt["src_core"]),
                dst=int(pkt["dst_core"]),
                size_bytes=int(pkt["size_bytes"]),
                payload_type=str(pkt.get("payload", "data")),
            )

            def _inject(p=p, d=delay, c=color):
                if d > 0:
                    yield self.env.timeout(d)
                yield self.env.process(self.send_packet(p, color=c))

            procs.append(self.env.process(_inject()))
        return procs

    def run_until_done(self, traffic: list[dict], color_for_payload: dict[str, int] | None = None) -> int:
        self.run_traffic(traffic, color_for_payload)
        self.env.run()
        return int(self.env.now)


def _mesh_manhattan(a: int, b: int, cols: int) -> int:
    ar, ac = divmod(a, cols)
    br, bc = divmod(b, cols)
    return abs(ar - br) + abs(ac - bc)

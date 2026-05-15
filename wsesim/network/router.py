"""Router model with configurable multi-stage pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import simpy


@dataclass(slots=True)
class Router:
    env: simpy.Environment
    node_id: int
    num_vcs: int
    buffer_depth: int
    pipeline_mode: str = "4_stage"
    routing_latency_cycles: int = 1
    vc_alloc_latency_cycles: int = 1
    switch_alloc_latency_cycles: int = 1
    switch_traversal_latency_cycles: int = 1
    crossbar_bw_flits_per_cycle: int = 1
    arbitration: str = "round_robin"
    input_buffer: simpy.Store = field(init=False)
    pipeline_unit: simpy.Resource = field(init=False)
    rc_unit: simpy.Resource = field(init=False)
    va_unit: simpy.Resource = field(init=False)
    sa_unit: simpy.Resource = field(init=False)
    st_unit: simpy.Resource = field(init=False)
    active_vc_packets: set[int] = field(init=False)

    def __post_init__(self) -> None:
        self.input_buffer = simpy.Store(self.env, capacity=self.num_vcs * self.buffer_depth)
        self.pipeline_unit = simpy.Resource(self.env, capacity=1)
        self.rc_unit = simpy.Resource(self.env, capacity=1)
        self.va_unit = simpy.Resource(self.env, capacity=1)
        self.sa_unit = simpy.Resource(self.env, capacity=1)
        self.st_unit = simpy.Resource(self.env, capacity=1)
        self.active_vc_packets = set()

    def enqueue(self, item):
        if len(self.input_buffer.items) >= self.input_buffer.capacity:
            raise BufferError(f"Router {self.node_id} input buffer overflow.")
        return self.input_buffer.put(item)

    def _route_compute(self):
        with self.rc_unit.request() as req:
            yield req
            yield self.env.timeout(self.routing_latency_cycles)

    def _vc_allocate(self):
        with self.va_unit.request() as req:
            yield req
            yield self.env.timeout(self.vc_alloc_latency_cycles)

    def _switch_allocate(self):
        with self.sa_unit.request() as req:
            yield req
            yield self.env.timeout(self.switch_alloc_latency_cycles)

    def _switch_traverse(self, flits: int):
        with self.st_unit.request() as req:
            yield req
            traversal_cycles = self.switch_traversal_latency_cycles + ceil(
                flits / max(self.crossbar_bw_flits_per_cycle, 1)
            )
            yield self.env.timeout(traversal_cycles)

    def pipeline(self, flits: int):
        # Model ingress dequeue from input buffer before entering pipeline stages.
        yield self.input_buffer.get()

        if self.pipeline_mode == "1_stage":
            # Compressed model: combine control stages into one stage.
            control_cycles = max(
                self.routing_latency_cycles,
                self.vc_alloc_latency_cycles,
                self.switch_alloc_latency_cycles,
                self.switch_traversal_latency_cycles,
            )
            total_cycles = control_cycles + ceil(flits / max(self.crossbar_bw_flits_per_cycle, 1))
            with self.pipeline_unit.request() as req:
                yield req
                yield self.env.timeout(total_cycles)
            return

        if self.pipeline_mode != "4_stage":
            raise ValueError(f"Unsupported pipeline mode: {self.pipeline_mode}")

        yield self.env.process(self._route_compute())
        yield self.env.process(self._vc_allocate())
        yield self.env.process(self._switch_allocate())
        yield self.env.process(self._switch_traverse(flits))

    def can_reserve_vc(self, packet_id: int) -> bool:
        if packet_id in self.active_vc_packets:
            return True
        return len(self.active_vc_packets) < self.num_vcs

    def reserve_vc(self, packet_id: int) -> bool:
        if packet_id in self.active_vc_packets:
            return True
        if len(self.active_vc_packets) >= self.num_vcs:
            return False
        self.active_vc_packets.add(packet_id)
        return True

    def release_vc(self, packet_id: int) -> None:
        self.active_vc_packets.discard(packet_id)

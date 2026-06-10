"""Router model with configurable multi-stage pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import simpy

from wsesim.network.tdm_clock import TDMClock


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
    tdm_clock: TDMClock | None = None
    color_count: int = 0
    input_buffer: simpy.Store = field(init=False)
    per_color_buffers: dict[int, simpy.Store] = field(init=False, default_factory=dict)
    pipeline_unit: simpy.Resource = field(init=False)
    rc_unit: simpy.Resource = field(init=False)
    va_unit: simpy.Resource = field(init=False)
    sa_unit: simpy.Resource = field(init=False)
    st_unit: simpy.Resource = field(init=False)
    active_vc_packets: set[int] = field(init=False)
    vc_wait_cycles: int = 0
    buffer_wait_cycles: int = 0
    color_buffer_wait_cycles: int = 0
    color_wait_cycles: dict[int, int] = field(default_factory=dict)
    pipeline_cycles: int = 0
    flits_processed: int = 0

    def __post_init__(self) -> None:
        if self.color_count > 0:
            self.input_buffer = simpy.Store(self.env, capacity=1)
            self.per_color_buffers = {
                color: simpy.Store(self.env, capacity=self.buffer_depth)
                for color in range(self.color_count)
            }
        else:
            self.input_buffer = simpy.Store(self.env, capacity=self.num_vcs * self.buffer_depth)
        self.pipeline_unit = simpy.Resource(self.env, capacity=1)
        self.rc_unit = simpy.Resource(self.env, capacity=1)
        self.va_unit = simpy.Resource(self.env, capacity=1)
        self.sa_unit = simpy.Resource(self.env, capacity=1)
        self.st_unit = simpy.Resource(self.env, capacity=1)
        self.active_vc_packets = set()

    def _resolve_color(self, color: int | None) -> int:
        if self.color_count <= 0:
            return 0
        if color is None or color < 0:
            return 0
        return int(color) % self.color_count

    def _uses_tdm_color(self, color: int | None) -> bool:
        return self.color_count > 0 and color is not None and color >= 0

    def can_admit(self, color: int | None = None) -> bool:
        if self.color_count <= 0:
            return len(self.input_buffer.items) < self.input_buffer.capacity
        idx = self._resolve_color(color)
        buf = self.per_color_buffers[idx]
        return len(buf.items) < buf.capacity

    def pending_for_color(self, color: int | None = None) -> int:
        if self.color_count <= 0:
            return len(self.input_buffer.items)
        idx = self._resolve_color(color)
        return len(self.per_color_buffers[idx].items)

    def capacity_for_color(self, color: int | None = None) -> int:
        if self.color_count <= 0:
            return self.input_buffer.capacity
        idx = self._resolve_color(color)
        return self.per_color_buffers[idx].capacity

    def enqueue(self, item):
        if self.color_count <= 0:
            if len(self.input_buffer.items) >= self.input_buffer.capacity:
                raise BufferError(f"Router {self.node_id} input buffer overflow.")
            return self.input_buffer.put(item)

        color = self._resolve_color(getattr(item, "color", None))
        raw = getattr(item, "color", None)
        if self.color_count > 0 and (raw is None or raw < 0):
            color = 0
        target = self.per_color_buffers[color] if self.color_count > 0 else self.input_buffer
        if len(target.items) >= target.capacity:
            raise BufferError(f"Router {self.node_id} color {color} buffer overflow.")
        return target.put(item)

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

    def pipeline(self, flits: int, color: int | None = None):
        start = self.env.now
        # Model ingress dequeue from input buffer before entering pipeline stages.
        if self.color_count > 0:
            color_idx = self._resolve_color(color)
            yield self.per_color_buffers[color_idx].get()
            if self._uses_tdm_color(color) and self.tdm_clock is not None:
                while self.tdm_clock.current_color(self.env.now) != color_idx:
                    self.color_buffer_wait_cycles += 1
                    self.color_wait_cycles[color_idx] = self.color_wait_cycles.get(color_idx, 0) + 1
                    yield self.env.timeout(1)
        else:
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
            self.pipeline_cycles += int(self.env.now - start)
            self.flits_processed += flits
            return

        if self.pipeline_mode != "4_stage":
            raise ValueError(f"Unsupported pipeline mode: {self.pipeline_mode}")

        yield self.env.process(self._route_compute())
        yield self.env.process(self._vc_allocate())
        yield self.env.process(self._switch_allocate())
        yield self.env.process(self._switch_traverse(flits))
        self.pipeline_cycles += int(self.env.now - start)
        self.flits_processed += flits

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

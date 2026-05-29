"""Per-color router with dedicated queues and static forwarding."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import simpy

from wsesim.network.color import ColorPlan
from wsesim.network.packet import Flit


@dataclass(slots=True)
class ColorRouter:
    env: simpy.Environment
    node_id: int
    plan: ColorPlan
    entries_per_color: int = 2
    pipeline_mode: str = "1_stage"
    routing_latency_cycles: int = 1
    vc_alloc_latency_cycles: int = 1
    switch_alloc_latency_cycles: int = 1
    switch_traversal_latency_cycles: int = 1
    crossbar_bw_flits_per_cycle: int = 1
    color_queues: dict[int, simpy.Store] = field(init=False)
    stall_out: dict[int, bool] = field(init=False)
    color_buffer_wait_cycles: int = 0
    pipeline_cycles: int = 0
    flits_processed: int = 0
    rr_index: int = 0

    def __post_init__(self) -> None:
        self.color_queues = {
            c: simpy.Store(self.env, capacity=self.entries_per_color)
            for c in range(self.plan.num_colors)
        }
        self.stall_out = {c: False for c in range(self.plan.num_colors)}

    def queue_len(self, color: int) -> int:
        return len(self.color_queues[color].items)

    def is_stalled(self, color: int) -> bool:
        return self.stall_out.get(color, False) or self.queue_len(color) >= self.entries_per_color

    def update_stall(self, color: int) -> None:
        self.stall_out[color] = self.queue_len(color) >= self.entries_per_color

    def enqueue(self, flit: Flit, color: int):
        if color not in self.color_queues:
            raise ValueError(f"Invalid color {color} at router {self.node_id}")
        return self.color_queues[color].put(flit)

    def pipeline(self, flits: int = 1):
        if self.pipeline_mode == "1_stage":
            control = max(
                self.routing_latency_cycles,
                self.vc_alloc_latency_cycles,
                self.switch_alloc_latency_cycles,
                self.switch_traversal_latency_cycles,
            )
            total = control + ceil(flits / max(self.crossbar_bw_flits_per_cycle, 1))
            start = self.env.now
            yield self.env.timeout(total)
            self.pipeline_cycles += int(self.env.now - start)
            self.flits_processed += flits
            return

        yield self.env.timeout(self.routing_latency_cycles)
        yield self.env.timeout(self.vc_alloc_latency_cycles)
        yield self.env.timeout(self.switch_alloc_latency_cycles)
        yield self.env.timeout(
            self.switch_traversal_latency_cycles
            + ceil(flits / max(self.crossbar_bw_flits_per_cycle, 1))
        )
        self.flits_processed += flits

    def pick_color_with_work(self) -> int | None:
        colors = list(range(self.plan.num_colors))
        for offset in range(len(colors)):
            c = colors[(self.rr_index + offset) % len(colors)]
            if self.color_queues[c].items and not self.is_stalled(c):
                self.rr_index = (c + 1) % len(colors)
                return c
        return None

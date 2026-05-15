"""Router model with simplified buffering."""

from __future__ import annotations

from dataclasses import dataclass, field

import simpy


@dataclass(slots=True)
class Router:
    env: simpy.Environment
    node_id: int
    num_vcs: int
    buffer_depth: int
    routing_latency_cycles: int = 1
    arbitration: str = "round_robin"
    input_buffer: simpy.Store = field(init=False)

    def __post_init__(self) -> None:
        self.input_buffer = simpy.Store(self.env, capacity=self.num_vcs * self.buffer_depth)

    def enqueue(self, item):
        if len(self.input_buffer.items) >= self.input_buffer.capacity:
            raise BufferError(f"Router {self.node_id} input buffer overflow.")
        return self.input_buffer.put(item)

    def route_compute(self):
        yield self.env.timeout(self.routing_latency_cycles)

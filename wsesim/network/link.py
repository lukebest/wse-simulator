"""Point-to-point link model with contention."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import simpy


@dataclass(slots=True)
class Link:
    env: simpy.Environment
    src: int
    dst: int
    bandwidth_flits_per_cycle: int
    latency_cycles: int
    resource: simpy.Resource = field(init=False)
    total_wait_cycles: int = 0
    total_busy_cycles: int = 0
    transfers: int = 0

    def __post_init__(self) -> None:
        self.resource = simpy.Resource(self.env, capacity=1)

    def transfer(self, flits: int):
        start_wait = self.env.now
        with self.resource.request() as req:
            yield req
            wait_cycles = int(self.env.now - start_wait)
            self.total_wait_cycles += wait_cycles

            tx_cycles = ceil(flits / max(self.bandwidth_flits_per_cycle, 1))
            busy_cycles = self.latency_cycles + tx_cycles
            self.total_busy_cycles += busy_cycles
            self.transfers += 1
            yield self.env.timeout(busy_cycles)

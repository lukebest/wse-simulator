"""Point-to-point link model."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import simpy


@dataclass(slots=True)
class Link:
    env: simpy.Environment
    src: int
    dst: int
    bandwidth_flits_per_cycle: int
    latency_cycles: int

    def transfer(self, flits: int):
        tx_cycles = ceil(flits / max(self.bandwidth_flits_per_cycle, 1))
        yield self.env.timeout(self.latency_cycles + tx_cycles)

"""Private L1 buffer model per core."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import simpy


@dataclass(slots=True)
class L1Buffer:
    env: simpy.Environment
    capacity_bytes: int
    read_bw_bytes_per_cycle: int
    write_bw_bytes_per_cycle: int
    access_latency_cycles: int = 1
    resource: simpy.Resource = field(init=False)

    def __post_init__(self) -> None:
        self.resource = simpy.Resource(self.env, capacity=1)

    def can_fit(self, size_bytes: int) -> bool:
        return size_bytes <= self.capacity_bytes

    def _transfer_cycles(self, size_bytes: int, bw_bytes_per_cycle: int) -> int:
        return self.access_latency_cycles + ceil(size_bytes / max(bw_bytes_per_cycle, 1))

    def read(self, size_bytes: int):
        with self.resource.request() as req:
            yield req
            yield self.env.timeout(
                self._transfer_cycles(size_bytes, self.read_bw_bytes_per_cycle)
            )

    def write(self, size_bytes: int):
        with self.resource.request() as req:
            yield req
            yield self.env.timeout(
                self._transfer_cycles(size_bytes, self.write_bw_bytes_per_cycle)
            )

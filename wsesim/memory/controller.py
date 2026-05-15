"""Memory controller with bandwidth contention."""

from __future__ import annotations

from dataclasses import dataclass, field

import simpy

from wsesim.memory.backend import MemoryBackend


@dataclass(slots=True)
class MemoryController:
    env: simpy.Environment
    backend: MemoryBackend
    max_concurrent: int = 1
    resource: simpy.Resource = field(init=False)

    def __post_init__(self) -> None:
        self.resource = simpy.Resource(self.env, capacity=self.max_concurrent)

    def access(self, size_bytes: int, is_write: bool = False):
        with self.resource.request() as req:
            yield req
            cycles = self.backend.request_cycles(size_bytes=size_bytes, is_write=is_write)
            yield self.env.timeout(cycles)

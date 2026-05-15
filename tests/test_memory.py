from __future__ import annotations

import simpy

from wsesim.memory.backend import AnalyticalMemoryBackend
from wsesim.memory.controller import MemoryController


def test_analytical_memory_access() -> None:
    env = simpy.Environment()
    backend = AnalyticalMemoryBackend(
        base_latency_cycles=10,
        bytes_per_cycle=128,
        jitter_model="none",
    )
    ctrl = MemoryController(env=env, backend=backend, max_concurrent=1)
    env.process(ctrl.access(size_bytes=1024))
    env.run()
    assert env.now >= 10

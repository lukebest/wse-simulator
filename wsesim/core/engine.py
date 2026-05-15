"""Core simulation engine wrapper around SimPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import simpy

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult


EventFn = Callable[[simpy.Environment], simpy.events.Event]


@dataclass(slots=True)
class SimulationEngine:
    config: WSEConfig
    env: simpy.Environment = field(default_factory=simpy.Environment)
    result: SimResult = field(default_factory=SimResult)

    def now(self) -> int | float:
        return self.env.now

    def timeout(self, cycles: int | float) -> simpy.events.Timeout:
        return self.env.timeout(cycles)

    def process(self, generator: EventFn) -> simpy.events.Process:
        return self.env.process(generator(self.env))

    def run(self, until: int | float | None = None) -> SimResult:
        self.env.run(until=until)
        return self.result

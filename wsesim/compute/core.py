"""Core model: PE + private L1 + network interface hooks."""

from __future__ import annotations

from dataclasses import dataclass

import simpy

from wsesim.compute.l1_buffer import L1Buffer
from wsesim.compute.pe import PEModel


@dataclass(slots=True)
class CoreTask:
    m: int
    n: int
    k: int
    a_bytes: int
    b_bytes: int
    c_bytes: int
    name: str = "matmul"
    use_double_buffer: bool = False


@dataclass(slots=True)
class Core:
    core_id: int
    env: simpy.Environment
    pe: PEModel
    l1: L1Buffer
    busy_cycles: int = 0

    def run_task(self, task: CoreTask):
        if not (self.l1.can_fit(task.a_bytes) and self.l1.can_fit(task.b_bytes)):
            raise ValueError("Task input tiles do not fit in private L1 buffer.")

        if task.use_double_buffer:
            # Simplified overlap model: hide one input read latency.
            read_a = self.l1.read(task.a_bytes)
            read_b = self.l1.read(task.b_bytes)
            yield self.env.process(read_a)
            compute_cycles = self.pe.matmul_cycles(task.m, task.n, task.k)
            yield self.env.process(read_b)
            yield self.env.timeout(compute_cycles)
        else:
            yield self.env.process(self.l1.read(task.a_bytes))
            yield self.env.process(self.l1.read(task.b_bytes))
            compute_cycles = self.pe.matmul_cycles(task.m, task.n, task.k)
            yield self.env.timeout(compute_cycles)

        self.busy_cycles += compute_cycles
        yield self.env.process(self.l1.write(task.c_bytes))

from __future__ import annotations

import simpy

from wsesim.compute.core import Core, CoreTask
from wsesim.compute.l1_buffer import L1Buffer
from wsesim.compute.pe import PEModel


def test_systolic_cycles_formula() -> None:
    pe = PEModel(pe_type="systolic", width=16)
    cycles = pe.matmul_cycles(m=32, n=32, k=64)
    assert cycles == (2 * 2 * 64) + 16 + 16 - 2


def test_core_runs_task() -> None:
    env = simpy.Environment()
    core = Core(
        core_id=0,
        env=env,
        pe=PEModel(pe_type="vector", width=16),
        l1=L1Buffer(
            env=env,
            capacity_bytes=1_000_000,
            read_bw_bytes_per_cycle=512,
            write_bw_bytes_per_cycle=512,
            access_latency_cycles=1,
        ),
    )
    task = CoreTask(m=16, n=16, k=16, a_bytes=1024, b_bytes=1024, c_bytes=1024)
    env.process(core.run_task(task))
    env.run()
    assert core.busy_cycles > 0

from __future__ import annotations

import simpy

from wsesim.network.packet import Flit, Packet
from wsesim.network.router import Router
from wsesim.network.tdm_clock import TDMClock


def _make_flit(color: int) -> Flit:
    pkt = Packet(src=0, dst=1, size_bytes=32, payload_type="allgather")
    return Flit(packet=pkt, flit_id=0, is_head=True, is_tail=True, color=color)


def test_router_uses_per_color_buffer_admission() -> None:
    env = simpy.Environment()
    router = Router(
        env=env,
        node_id=0,
        num_vcs=2,
        buffer_depth=1,
        tdm_clock=TDMClock(period=2, slot_cycles=1),
        color_count=2,
    )
    assert router.can_admit(1)
    env.run(until=router.enqueue(_make_flit(1)))
    assert not router.can_admit(1)
    assert router.can_admit(0)


def test_router_pipeline_waits_for_slot_match() -> None:
    env = simpy.Environment()
    router = Router(
        env=env,
        node_id=0,
        num_vcs=2,
        buffer_depth=2,
        tdm_clock=TDMClock(period=2, slot_cycles=4),
        color_count=2,
    )
    env.run(until=router.enqueue(_make_flit(1)))
    # t=0 in slot for color 0, color 1 opens at t=4.
    env.process(router.pipeline(1, color=1))
    env.run()
    assert env.now >= 4
    assert router.color_buffer_wait_cycles >= 4


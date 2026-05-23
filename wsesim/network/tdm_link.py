"""TDM-aware link model with global color cycles."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from wsesim.network.link import Link
from wsesim.network.tdm_clock import TDMClock


@dataclass(slots=True)
class TDMLink(Link):
    period: int = 1
    slot_cycles: int = 1
    tdm_clock: TDMClock | None = None
    active_logical_per_color: list[tuple[int, int] | None] | None = None

    def transfer(
        self,
        flits: int,
        flit_color: int | None = None,
        logical_link: tuple[int, int] | None = None,
    ):
        if self.active_logical_per_color is None or flit_color is None:
            yield from super(TDMLink, self).transfer(flits)
            return

        if flits != 1:
            raise ValueError("TDMLink only supports single-flit transfer per call.")

        start_wait = self.env.now
        spin_wait = 0
        while True:
            if self.tdm_clock is None:
                self.tdm_clock = TDMClock(period=max(1, self.period), slot_cycles=max(1, self.slot_cycles))
            cur_color = self.tdm_clock.current_color(self.env.now)
            active = self.active_logical_per_color[cur_color]
            color_match = cur_color == flit_color
            logical_match = logical_link is None or active == logical_link
            if color_match and logical_match and active is not None:
                break
            self.total_wait_cycles += 1
            spin_wait += 1
            yield self.env.timeout(1)

        with self.resource.request() as req:
            yield req
            wait_cycles = int(self.env.now - start_wait)
            # We already count per-cycle waits while spinning, only add queued wait here.
            queue_wait_cycles = max(0, wait_cycles - spin_wait)
            self.total_wait_cycles += queue_wait_cycles

            tx_cycles = ceil(flits / max(self.bandwidth_flits_per_cycle, 1))
            busy_cycles = self.latency_cycles + tx_cycles
            self.total_busy_cycles += busy_cycles
            self.transfers += 1
            yield self.env.timeout(busy_cycles)

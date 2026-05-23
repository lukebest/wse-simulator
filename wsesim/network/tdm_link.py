"""TDM-aware link model.

Color-slot gating is modeled only at the router (see ``Router.pipeline``).
Physical links carry flits once the router releases them; ``active_logical_per_color``
is retained for compile-time bookkeeping / diagnostics but is not re-checked at transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        del flit_color, logical_link
        if flits != 1:
            raise ValueError("TDMLink only supports single-flit transfer per call.")
        yield from super().transfer(flits)

"""Per-color credit / backpressure flow control."""

from __future__ import annotations

from wsesim.network.flow_control.base import FlowControl


class PerColorCreditFlowControl(FlowControl):
    """Patent-style per-color backpressure: stall when downstream queue full."""

    def __init__(self, entries_per_color: int = 2) -> None:
        self.entries_per_color = entries_per_color

    def can_send(self, downstream_queue_len: int, downstream_capacity: int) -> bool:
        return downstream_queue_len < min(downstream_capacity, self.entries_per_color)

    def capacity_for_color(self, num_colors: int, entries_per_color: int | None = None) -> int:
        e = entries_per_color if entries_per_color is not None else self.entries_per_color
        return max(1, e)

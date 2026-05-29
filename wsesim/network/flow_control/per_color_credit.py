"""Per-color credit / backpressure flow control."""

from __future__ import annotations

from wsesim.network.flow_control.base import FlowControl


class PerColorCreditFlowControl(FlowControl):
    """Credit-style: send when downstream per-color queue has space."""

    def __init__(self, entries_per_color: int) -> None:
        self.entries_per_color = entries_per_color

    def can_send(self, downstream_queue_len: int, downstream_capacity: int) -> bool:
        del downstream_capacity
        return downstream_queue_len < self.entries_per_color

    def is_stalled(self, queue_len: int) -> bool:
        return queue_len >= self.entries_per_color

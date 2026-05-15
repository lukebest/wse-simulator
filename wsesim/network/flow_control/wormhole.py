"""Wormhole-style flow control (simplified)."""

from __future__ import annotations

from wsesim.network.flow_control.base import FlowControl


class WormholeFlowControl(FlowControl):
    def can_send(self, downstream_queue_len: int, downstream_capacity: int) -> bool:
        # Conservative policy to reduce contention under wormhole assumptions.
        return downstream_queue_len < max(1, downstream_capacity - 1)

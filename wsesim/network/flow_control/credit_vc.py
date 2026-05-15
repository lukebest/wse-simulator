"""Credit-based virtual-channel flow control."""

from __future__ import annotations

from wsesim.network.flow_control.base import FlowControl


class CreditBasedVCFlowControl(FlowControl):
    def can_send(self, downstream_queue_len: int, downstream_capacity: int) -> bool:
        return downstream_queue_len < downstream_capacity

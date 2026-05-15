"""Flow-control interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class FlowControl(ABC):
    @abstractmethod
    def can_send(self, downstream_queue_len: int, downstream_capacity: int) -> bool:
        raise NotImplementedError

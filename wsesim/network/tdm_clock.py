"""Shared TDM clock utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TDMClock:
    period: int
    slot_cycles: int = 1

    def current_color(self, now: float) -> int:
        p = max(1, int(self.period))
        slot = max(1, int(self.slot_cycles))
        return (int(now) // slot) % p

    def cycles_until_color(self, now: float, target_color: int) -> int:
        p = max(1, int(self.period))
        slot = max(1, int(self.slot_cycles))
        target = int(target_color) % p
        cur = int(now)
        current_color = self.current_color(cur)
        if current_color == target:
            return 0
        current_slot = cur // slot
        delta_slots = (target - current_color) % p
        next_slot_start = (current_slot + delta_slots) * slot
        return max(0, next_slot_start - cur)

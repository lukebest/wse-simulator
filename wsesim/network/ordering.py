"""Flit ordering tracker for per-color FIFO guarantees."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OrderTracker:
    """Track per-packet flit order and per-stream packet order."""

    expected_flit: dict[tuple[int, int, int], int] = field(default_factory=dict)
    expected_pkt: dict[tuple[int, int], int] = field(default_factory=dict)
    violations: int = 0
    deliveries: list[tuple[int, int, int, float]] = field(default_factory=list)

    def record_flit_delivery(self, src: int, color: int, packet_id: int, flit_id: int, time: float) -> None:
        key = (src, color, packet_id)
        expected = self.expected_flit.get(key, 0)
        if flit_id != expected:
            self.violations += 1
        self.expected_flit[key] = expected + 1
        self.deliveries.append((src, color, flit_id, time))

    def record_delivery(self, src: int, color: int, seq: int, time: float) -> None:
        """Record packet-level delivery order for a (src, color) stream."""
        key = (src, color)
        expected = self.expected_pkt.get(key, 0)
        if seq != expected:
            self.violations += 1
        self.expected_pkt[key] = expected + 1
        self.deliveries.append((src, color, seq, time))

    def assert_no_violations(self) -> None:
        if self.violations > 0:
            raise AssertionError(f"Ordering violations: {self.violations}")

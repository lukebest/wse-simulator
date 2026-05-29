"""Color-aware router with per-color queues and multicast replication."""

from __future__ import annotations

from dataclasses import dataclass, field

import simpy

from wsesim.network.color import ColorPlan


@dataclass(slots=True)
class ColorRouter:
    """Patent Router 600 simplified: per-color input queues + per-direction scheduler."""

    env: simpy.Environment
    node_id: int
    num_colors: int
    entries_per_color: int = 2
    pipeline_cycles: int = 1
    color_queues: dict[int, simpy.Store] = field(init=False)
    color_buffer_wait_cycles: int = 0
    pipeline_cycles_total: int = 0
    flits_processed: int = 0
    sent_bitmap: dict[int, set[int]] = field(init=False)  # color -> directions sent (for multicast tracking)
    rr_cursor: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        cap = max(1, self.entries_per_color)
        self.color_queues = {
            c: simpy.Store(self.env, capacity=cap) for c in range(self.num_colors)
        }
        self.sent_bitmap = {c: set() for c in range(self.num_colors)}
        self.rr_cursor = {c: 0 for c in range(self.num_colors)}

    def can_accept(self, color_id: int) -> bool:
        q = self.color_queues[color_id]
        return len(q.items) < q.capacity

    def enqueue(self, color_id: int, item):
        if not self.can_accept(color_id):
            raise BufferError(f"Color {color_id} queue full at router {self.node_id}")
        return self.color_queues[color_id].put(item)

    def is_stalled(self, color_id: int) -> bool:
        return not self.can_accept(color_id)

    def dequeue(self, color_id: int):
        return self.color_queues[color_id].get()

    def reset_multicast_state(self, color_id: int) -> None:
        self.sent_bitmap[color_id] = set()

    def mark_sent(self, color_id: int, direction_node: int) -> None:
        self.sent_bitmap[color_id].add(direction_node)

    def already_sent(self, color_id: int, direction_node: int) -> bool:
        return direction_node in self.sent_bitmap[color_id]

    def pipeline(self, flits: int = 1):
        start = self.env.now
        yield self.env.timeout(self.pipeline_cycles * flits)
        self.pipeline_cycles_total += int(self.env.now - start)
        self.flits_processed += flits

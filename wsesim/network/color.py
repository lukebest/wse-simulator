"""Color virtual-network data model (patent US10,515,303)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Color:
    """A virtual network identifier with optional task binding."""

    color_id: int
    task_id: int = 0

    def instruction_offset(self) -> int:
        """Patent: base + color * 4 selects handler task."""
        return self.color_id * 4


@dataclass(slots=True)
class ColorPlan:
    """Static per-node per-color forwarding table (Dest 661 semantics).

    dest[node][color_id] -> frozenset of next-hop node ids.
    Empty set means no forwarding at this node for that color.
    Multiple next hops imply in-router multicast replication.
    """

    num_colors: int
    dest: dict[int, dict[int, frozenset[int]]] = field(default_factory=dict)
    color_tasks: dict[int, int] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def next_hops(self, node: int, color_id: int) -> frozenset[int]:
        return self.dest.get(node, {}).get(color_id, frozenset())

    def set_next_hops(self, node: int, color_id: int, hops: frozenset[int]) -> None:
        self.dest.setdefault(node, {})[color_id] = hops

    def trace_route(self, src: int, dst: int, color_id: int, max_hops: int = 10_000) -> list[int] | None:
        """Follow fixed color forwarding from src; succeed if dst reached."""
        if src == dst:
            return [src]
        path = [src]
        current = src
        for _ in range(max_hops):
            hops = self.next_hops(current, color_id)
            if not hops:
                return None
            if dst in hops:
                path.append(dst)
                return path
            if len(hops) > 1:
                nxt = min(hops, key=lambda h: abs(h - dst))
            else:
                nxt = next(iter(hops))
            if nxt in path:
                return None
            path.append(nxt)
            current = nxt
            if current == dst:
                return path
        return None

    def merge(self, other: ColorPlan) -> None:
        """Merge another plan's dest entries (must not conflict)."""
        for node, colors in other.dest.items():
            for cid, hops in colors.items():
                existing = self.next_hops(node, cid)
                if existing and existing != hops:
                    raise ValueError(f"Color {cid} conflict at node {node}: {existing} vs {hops}")
                self.set_next_hops(node, cid, hops)
        self.color_tasks.update(other.color_tasks)

    @staticmethod
    def empty(num_nodes: int, num_colors: int) -> ColorPlan:
        plan = ColorPlan(num_colors=num_colors)
        for node in range(num_nodes):
            plan.dest[node] = {c: frozenset() for c in range(num_colors)}
        return plan

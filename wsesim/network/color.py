"""Color virtual-network data model (US10,515,303)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True, frozen=True)
class Color:
    """Virtual network identifier and optional task selector."""

    color_id: int
    task: int = 0
    name: str = ""


@dataclass(slots=True)
class ColorPlan:
    """Static per-node per-color forwarding (Dest bit-vector as neighbor sets)."""

    num_colors: int
    rows: int
    cols: int
    colors: list[Color] = field(default_factory=list)
    # node -> color_id -> set of next-hop node ids (empty = hold/consume at CE)
    dest: dict[int, dict[int, set[int]]] = field(default_factory=dict)
    # Unicast fallback: color_id -> routing mode name for dst-dependent static rules
    unicast_modes: dict[int, str] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return self.rows * self.cols

    def ensure_node(self, node: int) -> dict[int, set[int]]:
        if node not in self.dest:
            self.dest[node] = {c: set() for c in range(self.num_colors)}
        return self.dest[node]

    def set_dest(self, node: int, color_id: int, next_hops: set[int]) -> None:
        self.ensure_node(node)[color_id] = set(next_hops)

    def set_unicast_mode(self, color_id: int, mode: str) -> None:
        self.unicast_modes[color_id] = mode

    def next_hops(self, node: int, color_id: int, dst: int | None = None) -> set[int]:
        """Return static next hops for (node, color). Uses unicast mode if dest empty."""
        table = self.dest.get(node, {})
        hops = table.get(color_id, set())
        if hops:
            return hops
        if dst is not None and color_id in self.unicast_modes:
            nxt = _unicast_step(node, dst, self.unicast_modes[color_id], self.rows, self.cols)
            if nxt is not None and nxt != node:
                return {nxt}
        return set()

    def iter_colors(self) -> Iterator[Color]:
        if self.colors:
            yield from self.colors
        else:
            for cid in range(self.num_colors):
                yield Color(color_id=cid, task=cid * 4)


def _unicast_step(
    current: int, dst: int, mode: str, rows: int, cols: int
) -> int | None:
    if current == dst:
        return current
    r, c = divmod(current, cols)
    dr, dc = divmod(dst, cols)
    if mode == "xy":
        if r < dr:
            return (r + 1) * cols + c
        if r > dr:
            return (r - 1) * cols + c
        if c < dc:
            return r * cols + (c + 1)
        if c > dc:
            return r * cols + (c - 1)
    elif mode == "yx":
        if c < dc:
            return r * cols + (c + 1)
        if c > dc:
            return r * cols + (c - 1)
        if r < dr:
            return (r + 1) * cols + c
        if r > dr:
            return (r - 1) * cols + c
    return None

"""End-to-end color assignment for TDM flattened butterfly.

Semantics (per user):
- Each (src, dst) communication picks ONE color up-front and uses it for the
  whole lifetime of the transfer.
- Within a single color, all simultaneously-active (src, dst) paths must be
  physically edge-disjoint on the underlying mesh.
- Routing uses dimension-order FB (changes one logical dim per logical hop),
  whose physical embedding equals XY routing on the mesh. Max logical hops = n.

The resulting plan therefore assigns ONE color per directed (src, dst) pair.
"""

from __future__ import annotations

from dataclasses import dataclass


PhysicalEdge = tuple[int, int]
Pair = tuple[int, int]


@dataclass(slots=True)
class E2EColorPlan:
    C: int
    color_lower_bound: int
    color_of_pair: dict[Pair, int]
    pairs_by_color: dict[int, list[Pair]]
    path_of_pair: dict[Pair, list[PhysicalEdge]]
    load_per_physical_link: dict[PhysicalEdge, int]


def physical_xy_path(src: int, dst: int, rows: int, cols: int) -> list[PhysicalEdge]:
    if src == dst:
        return []
    path: list[PhysicalEdge] = []
    sr, sc = divmod(src, cols)
    dr, dc = divmod(dst, cols)
    c = sc
    while c != dc:
        nc = c + (1 if dc > c else -1)
        path.append((sr * cols + c, sr * cols + nc))
        c = nc
    r = sr
    while r != dr:
        nr = r + (1 if dr > r else -1)
        path.append((r * cols + dc, nr * cols + dc))
        r = nr
    return path


def assign_e2e_colors(rows: int, cols: int) -> E2EColorPlan:
    num_nodes = rows * cols
    pairs: list[Pair] = [
        (s, d) for s in range(num_nodes) for d in range(num_nodes) if s != d
    ]
    paths: dict[Pair, list[PhysicalEdge]] = {
        pair: physical_xy_path(pair[0], pair[1], rows, cols) for pair in pairs
    }

    load: dict[PhysicalEdge, int] = {}
    for pair, path in paths.items():
        for edge in path:
            load[edge] = load.get(edge, 0) + 1
    lower_bound = max(load.values()) if load else 1

    # Greedy: assign hardest (longest) paths first; tie-break by edge contention.
    def pair_priority(pair: Pair) -> tuple[int, int]:
        path = paths[pair]
        contention = sum(load.get(edge, 0) for edge in path)
        return (-len(path), -contention)

    color_count = lower_bound
    free_at: dict[PhysicalEdge, list[bool]] = {
        edge: [True] * color_count for edge in load
    }
    color_of: dict[Pair, int] = {}

    for pair in sorted(pairs, key=pair_priority):
        path = paths[pair]
        if not path:
            continue
        chosen: int | None = None
        for color in range(color_count):
            if all(free_at.setdefault(edge, [True] * color_count)[color] for edge in path):
                chosen = color
                break
        if chosen is None:
            for edge in free_at:
                free_at[edge].append(True)
            color_count += 1
            chosen = color_count - 1
            for edge in path:
                free_at.setdefault(edge, [True] * color_count)
                free_at[edge][chosen] = False
        else:
            for edge in path:
                free_at[edge][chosen] = False
        color_of[pair] = chosen

    pairs_by_color: dict[int, list[Pair]] = {c: [] for c in range(color_count)}
    for pair, color in color_of.items():
        pairs_by_color[color].append(pair)

    return E2EColorPlan(
        C=color_count,
        color_lower_bound=lower_bound,
        color_of_pair=color_of,
        pairs_by_color=pairs_by_color,
        path_of_pair=paths,
        load_per_physical_link=load,
    )

"""Static color route builders for 2D mesh."""

from __future__ import annotations

from collections import deque
from math import ceil, log2

from wsesim.network.color import ColorPlan


def mesh_dims(num_nodes: int, cols: int | None = None) -> tuple[int, int]:
    if cols is not None:
        rows = num_nodes // cols
        if rows * cols != num_nodes:
            raise ValueError("rows*cols must equal num_nodes")
        return rows, cols
    side = int(num_nodes**0.5)
    if side * side != num_nodes:
        raise ValueError("num_nodes must be square if cols not given")
    return side, side


def _node_rc(node: int, cols: int) -> tuple[int, int]:
    return divmod(node, cols)


def _rc_node(r: int, c: int, cols: int) -> int:
    return r * cols + c


def xy_path(src: int, dst: int, rows: int, cols: int) -> list[int]:
    """Dimension-order XY path on mesh."""
    sr, sc = _node_rc(src, cols)
    dr, dc = _node_rc(dst, cols)
    path = [src]
    r, c = sr, sc
    while c != dc:
        c += 1 if dc > c else -1
        path.append(_rc_node(r, c, cols))
    while r != dr:
        r += 1 if dr > r else -1
        path.append(_rc_node(r, c, cols))
    return path


def build_path_color(
    plan: ColorPlan,
    color_id: int,
    src: int,
    dst: int,
    rows: int,
    cols: int,
    *,
    order: str = "xy",
) -> None:
    """1→1 path color along XY (or YX) route."""
    if order == "yx":
        sr, sc = _node_rc(src, cols)
        dr, dc = _node_rc(dst, cols)
        path = [src]
        r, c = sr, sc
        while r != dr:
            r += 1 if dr > r else -1
            path.append(_rc_node(r, c, cols))
        while c != dc:
            c += 1 if dc > c else -1
            path.append(_rc_node(r, c, cols))
    else:
        path = xy_path(src, dst, rows, cols)
    for i in range(len(path) - 1):
        plan.set_next_hops(path[i], color_id, frozenset({path[i + 1]}))


def build_ring_color(
    plan: ColorPlan,
    color_id: int,
    nodes: list[int],
) -> None:
    """Hamiltonian ring: node[i] -> node[(i+1) % len]."""
    if len(nodes) < 2:
        return
    for i, node in enumerate(nodes):
        nxt = nodes[(i + 1) % len(nodes)]
        plan.set_next_hops(node, color_id, frozenset({nxt}))


def build_snake_ring_color(
    plan: ColorPlan,
    color_id: int,
    rows: int,
    cols: int,
) -> None:
    """Serpentine Hamiltonian ring over full mesh."""
    nodes: list[int] = []
    for r in range(rows):
        row_nodes = [_rc_node(r, c, cols) for c in range(cols)]
        if r % 2 == 1:
            row_nodes.reverse()
        nodes.extend(row_nodes)
    build_ring_color(plan, color_id, nodes)


def build_multicast_tree_color(
    plan: ColorPlan,
    color_id: int,
    root: int,
    targets: list[int],
    rows: int,
    cols: int,
) -> None:
    """1→N row-wise multicast tree from root."""
    root_r, root_c = _node_rc(root, cols)
    for tgt in targets:
        if tgt == root:
            continue
        tr, tc = _node_rc(tgt, cols)
        if tr != root_r:
            continue
        path = xy_path(root, tgt, rows, cols)
        for i in range(len(path) - 1):
            plan.set_next_hops(path[i], color_id, frozenset({path[i + 1]}))


def build_reduction_tree_color(
    plan: ColorPlan,
    color_id: int,
    leaves: list[int],
    root: int,
    rows: int,
    cols: int,
) -> None:
    """N→1 reduction tree converging at root (reverse path forwarding)."""
    for leaf in leaves:
        if leaf == root:
            continue
        path = xy_path(leaf, root, rows, cols)
        for i in range(len(path) - 1):
            plan.set_next_hops(path[i], color_id, frozenset({path[i + 1]}))


def build_row_bus_color(
    plan: ColorPlan,
    color_id: int,
    row: int,
    cols: int,
    direction: str = "east",
) -> None:
    """Per-row X-bus: each node forwards to neighbor in row."""
    nodes = [_rc_node(row, c, cols) for c in range(cols)]
    if direction == "west":
        nodes = list(reversed(nodes))
    for i in range(len(nodes) - 1):
        plan.set_next_hops(nodes[i], color_id, frozenset({nodes[i + 1]}))


def build_col_bus_color(
    plan: ColorPlan,
    color_id: int,
    col: int,
    rows: int,
    cols: int,
    direction: str = "south",
) -> None:
    """Per-column Y-bus."""
    nodes = [_rc_node(r, col, cols) for r in range(rows)]
    if direction == "north":
        nodes = list(reversed(nodes))
    for i in range(len(nodes) - 1):
        plan.set_next_hops(nodes[i], color_id, frozenset({nodes[i + 1]}))


def build_dimension_exchange_colors(
    plan: ColorPlan,
    nodes: list[int],
    base_color: int = 0,
) -> int:
    """Butterfly/RHD stage colors; returns next free color id."""
    s = len(nodes)
    if s <= 1:
        return base_color
    stages = int(ceil(log2(s))) if s > 1 else 0
    color = base_color
    for stage in range(stages):
        stride = 1 << stage
        for idx in range(s):
            partner_idx = idx ^ stride
            if partner_idx >= s:
                continue
            src, dst = nodes[idx], nodes[partner_idx]
            plan.set_next_hops(src, color, frozenset({dst}))
            plan.set_next_hops(dst, color, frozenset({src}))
        color += 1
    return color


def build_all_to_all_phase_colors(
    plan: ColorPlan,
    nodes: list[int],
    base_color: int = 0,
) -> int:
    """Time-phased permutation colors for all-to-all (Latin-square style)."""
    s = len(nodes)
    color = base_color
    for phase in range(s):
        for i in range(s):
            src = nodes[i]
            dst = nodes[(i + phase) % s]
            if src == dst:
                continue
            # Use single-hop if neighbors else store one step toward dst
            plan.set_next_hops(src, color, frozenset({dst}))
        color += 1
    return color


def build_column_reduction_colors(
    plan: ColorPlan,
    rows: int,
    cols: int,
    base_color: int = 0,
) -> int:
    """One reduction-tree color per column."""
    color = base_color
    for c in range(cols):
        leaves = [_rc_node(r, c, cols) for r in range(rows - 1, 0, -1)]
        root = _rc_node(0, c, cols)
        build_reduction_tree_color(plan, color, leaves, root, rows, cols)
        color += 1
    return color


def build_row_broadcast_colors(
    plan: ColorPlan,
    rows: int,
    cols: int,
    base_color: int = 0,
) -> int:
    """One multicast-tree color per row from westmost PE."""
    color = base_color
    for r in range(rows):
        root = _rc_node(r, 0, cols)
        targets = [_rc_node(r, c, cols) for c in range(1, cols)]
        build_multicast_tree_color(plan, color, root, targets, rows, cols)
        color += 1
    return color


def build_default_mixed_plan(num_nodes: int, num_colors: int = 16, cols: int | None = None) -> ColorPlan:
    """Build a mixed-workload color plan using available colors."""
    rows, cols = mesh_dims(num_nodes, cols)
    plan = ColorPlan.empty(num_nodes, num_colors)
    color = 0
    build_snake_ring_color(plan, color, rows, cols)
    color += 1
    color = build_column_reduction_colors(plan, rows, cols, color)
    color = build_row_broadcast_colors(plan, rows, cols, color)
    nodes = list(range(num_nodes))
    if color < num_colors:
        color = build_dimension_exchange_colors(plan, nodes[: min(len(nodes), 16)], color)
    return plan


def is_acyclic_color(plan: ColorPlan, color_id: int) -> bool:
    """Check if forwarding graph for one color is acyclic (deadlock-safe for trees/paths)."""
    # Build adjacency
    adj: dict[int, list[int]] = {}
    for node, colors in plan.dest.items():
        hops = colors.get(color_id, frozenset())
        if hops:
            adj[node] = list(hops)
    # DFS cycle detection
    visited: set[int] = set()
    stack: set[int] = set()

    def dfs(n: int) -> bool:
        visited.add(n)
        stack.add(n)
        for nb in adj.get(n, []):
            if nb not in visited:
                if not dfs(nb):
                    return False
            elif nb in stack:
                return False
        stack.remove(n)
        return True

    for n in adj:
        if n not in visited and not dfs(n):
            return False
    return True

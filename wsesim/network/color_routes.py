"""Static color route builders for 2D mesh virtual networks."""

from __future__ import annotations

from math import ceil, log2

from wsesim.network.color import Color, ColorPlan


def mesh_dims(num_nodes: int) -> tuple[int, int]:
    side = int(num_nodes**0.5)
    if side * side != num_nodes:
        raise ValueError(f"{num_nodes} is not a perfect square for mesh2d.")
    return side, side


def empty_plan(rows: int, cols: int, num_colors: int) -> ColorPlan:
    return ColorPlan(num_colors=num_colors, rows=rows, cols=cols)


def add_path_color(plan: ColorPlan, color_id: int, mode: str = "xy", name: str = "") -> None:
    plan.set_unicast_mode(color_id, mode)
    if color_id >= len(plan.colors):
        plan.colors.append(Color(color_id=color_id, task=color_id * 4, name=name or f"path_{mode}"))


def add_row_ring(plan: ColorPlan, color_id: int, row: int, name: str = "") -> None:
    cols = plan.cols
    nodes = [row * cols + c for c in range(cols)]
    for idx, node in enumerate(nodes):
        plan.set_dest(node, color_id, {nodes[(idx + 1) % cols]})
    _append_color(plan, color_id, name or f"row_ring_{row}")


def add_col_ring(plan: ColorPlan, color_id: int, col: int, name: str = "") -> None:
    cols = plan.cols
    rows = plan.rows
    nodes = [r * cols + col for r in range(rows)]
    for idx, node in enumerate(nodes):
        plan.set_dest(node, color_id, {nodes[(idx + 1) % rows]})
    _append_color(plan, color_id, name or f"col_ring_{col}")


def add_linear_ring(plan: ColorPlan, color_id: int, nodes: list[int] | None = None) -> None:
    """Ring following node-id order (matches ring collective traffic)."""
    order = nodes or list(range(plan.num_nodes))
    for idx, node in enumerate(order):
        plan.set_dest(node, color_id, {order[(idx + 1) % len(order)]})


def add_snake_ring(plan: ColorPlan, color_id: int, name: str = "snake_ring") -> None:
    cols = plan.cols
    rows = plan.rows
    order: list[int] = []
    for r in range(rows):
        row_nodes = [r * cols + c for c in (range(cols) if r % 2 == 0 else range(cols - 1, -1, -1))]
        order.extend(row_nodes)
    for idx, node in enumerate(order):
        plan.set_dest(node, color_id, {order[(idx + 1) % len(order)]})
    _append_color(plan, color_id, name)


def add_row_multicast_tree(plan: ColorPlan, color_id: int, row: int, root_col: int = 0) -> None:
    """Broadcast along row: root forwards to right; interior forwards right."""
    cols = plan.cols
    root = row * cols + root_col
    for c in range(cols):
        node = row * cols + c
        if c < cols - 1:
            plan.set_dest(node, color_id, {row * cols + (c + 1)})
        else:
            plan.set_dest(node, color_id, set())
    _append_color(plan, color_id, f"row_mcast_{row}")


def add_col_reduction_tree(plan: ColorPlan, color_id: int, col: int, root_row: int = 0) -> None:
    """Reduce toward top of column: each node sends north."""
    cols = plan.cols
    rows = plan.rows
    for r in range(rows):
        node = r * cols + col
        if r > root_row:
            plan.set_dest(node, color_id, {(r - 1) * cols + col})
        else:
            plan.set_dest(node, color_id, set())
    _append_color(plan, color_id, f"col_reduce_{col}")


def add_row_bus(plan: ColorPlan, color_id: int, row: int) -> None:
    add_row_multicast_tree(plan, color_id, row)


def add_col_bus(plan: ColorPlan, color_id: int, col: int) -> None:
    add_col_reduction_tree(plan, color_id, col)


def add_dimension_exchange_colors(
    plan: ColorPlan, start_color: int, nodes: list[int]
) -> list[int]:
    """One color per RHD stage (XOR partner exchange along snake order index)."""
    s = len(nodes)
    if s <= 1:
        return []
    stages = int(log2(s)) if _is_pow2(s) else 0
    if stages == 0:
        add_path_color(plan, start_color, "xy", "dim_ex_fallback")
        return [start_color]
    used: list[int] = []
    index_of = {n: i for i, n in enumerate(nodes)}
    for stage in range(stages):
        cid = start_color + stage
        stride = 1 << stage
        for node in nodes:
            idx = index_of[node]
            partner_idx = idx ^ stride
            if partner_idx < s:
                plan.set_dest(node, cid, {nodes[partner_idx]})
            else:
                plan.set_dest(node, cid, set())
        _append_color(plan, cid, f"dim_ex_s{stage}")
        used.append(cid)
    return used


def add_all_to_all_phases(plan: ColorPlan, start_color: int, nodes: list[int]) -> list[int]:
    """Time-phased colors: each phase is a fixed permutation (rotate by phase)."""
    s = len(nodes)
    if s <= 1:
        return []
    used: list[int] = []
    for phase in range(s - 1):
        cid = start_color + phase
        for idx, src in enumerate(nodes):
            dst = nodes[(idx + phase + 1) % s]
            if dst != src:
                plan.set_unicast_mode(cid, "xy")
                # Per-phase unicast: store explicit dest at src only via mode override
                # Use dest table at src pointing toward dst for this phase
                _set_unicast_path(plan, cid, src, dst)
        _append_color(plan, cid, f"a2a_p{phase}")
        used.append(cid)
    return used


def _set_unicast_path(plan: ColorPlan, color_id: int, src: int, dst: int) -> None:
    """Install hop-by-hop dest entries along XY path for one (src,dst) on a color."""
    current = src
    while current != dst:
        r, c = divmod(current, plan.cols)
        dr, dc = divmod(dst, plan.cols)
        if r < dr:
            nxt = (r + 1) * plan.cols + c
        elif r > dr:
            nxt = (r - 1) * plan.cols + c
        elif c < dc:
            nxt = r * plan.cols + (c + 1)
        elif c > dc:
            nxt = r * plan.cols + (c - 1)
        else:
            break
        plan.set_dest(current, color_id, {nxt})
        current = nxt


def build_mixed_plan(rows: int, cols: int, num_colors: int = 16) -> ColorPlan:
    """Default K-color plan covering NN communication taxonomy on mesh."""
    plan = empty_plan(rows, cols, num_colors)
    cid = 0
    add_path_color(plan, cid, "xy", "systolic_xy")
    cid += 1
    add_path_color(plan, cid, "yx", "systolic_yx")
    cid += 1
    add_linear_ring(plan, cid)
    _append_color(plan, cid, "allreduce_ring")
    cid += 1
    if rows > 0:
        add_row_multicast_tree(plan, cid, row=0)
        cid += 1
    if cols > 0:
        add_col_reduction_tree(plan, cid, col=0)
        cid += 1
    nodes = list(range(rows * cols))
    n = min(len(nodes), 1 << int(log2(max(2, len(nodes)))))
    pow2_nodes = nodes[:n] if _is_pow2(n) else nodes[: 1 << int(log2(len(nodes)))]
    if pow2_nodes and cid < num_colors:
        used = add_dimension_exchange_colors(plan, cid, pow2_nodes)
        cid += len(used)
    while cid < num_colors and cid < rows:
        add_row_ring(plan, cid, row=cid % rows)
        cid += 1
    while cid < num_colors and (cid - rows) < cols:
        add_col_ring(plan, cid, col=(cid - rows) % cols)
        cid += 1
    while cid < num_colors:
        add_path_color(plan, cid, "xy" if cid % 2 == 0 else "yx", f"spare_{cid}")
        cid += 1
    return plan


def build_baseline_single_vn(rows: int, cols: int) -> ColorPlan:
    plan = empty_plan(rows, cols, 1)
    add_path_color(plan, 0, "xy", "single_vn_xy")
    return plan


def _append_color(plan: ColorPlan, color_id: int, name: str) -> None:
    while len(plan.colors) <= color_id:
        plan.colors.append(Color(color_id=len(plan.colors), task=len(plan.colors) * 4))
    plan.colors[color_id] = Color(color_id=color_id, task=color_id * 4, name=name)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

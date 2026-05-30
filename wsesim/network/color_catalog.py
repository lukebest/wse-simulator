"""Mesh-size-independent color catalog (US10,515,303).

Each color encodes ONE static directional forwarding rule on a 2D mesh. The
catalog -- the set of color IDs and their semantics -- is identical for any
mesh size; only the concrete ``dest`` tables differ because they are
instantiated for the given ``(rows, cols)``. This realises the patent's
"compile-time static routing" where a color's route does not depend on the
runtime packet destination, only on the node it currently sits at.

A color whose ``dest[node]`` points one hop in a fixed compass direction acts
as a *directional bus*: a packet rides it until it reaches its destination
(the network stops forwarding once ``current == dst``). Collectives are then
realised by choosing which bus each transfer rides plus level-based delays.

Up to 24 colors are defined (the patent supports 16/24/32).
"""

from __future__ import annotations

from wsesim.network.color import Color, ColorPlan

MAX_COLORS = 24

# --- Fixed semantic color IDs (independent of mesh scale) ---
C_UNICAST_XY = 0
C_UNICAST_YX = 1
C_BCAST_ROW_EAST = 2     # broadcast: spread along a row, west->east
C_BCAST_COL_SOUTH = 3    # broadcast: spread down columns, north->south
C_GATHER_COL_NORTH = 4   # gather: converge up columns toward row 0
C_GATHER_ROW_WEST = 5    # gather: converge along row 0 toward col 0
C_REDUCE_COL_NORTH = 6   # reduce: same shape as gather, separate VN
C_REDUCE_ROW_WEST = 7
C_AG_ROW_EAST = 8        # allgather: row ring, eastbound
C_AG_ROW_WEST = 9        # allgather: row ring, westbound
C_AG_COL_SOUTH = 10      # allgather: column ring, southbound
C_AG_COL_NORTH = 11      # allgather: column ring, northbound
C_AR_RS_ROW_EAST = 12    # allreduce reduce-scatter along rows
C_AR_AG_ROW_WEST = 13    # allreduce allgather along rows
C_AR_RS_COL_SOUTH = 14   # allreduce reduce-scatter along columns
C_AR_AG_COL_NORTH = 15   # allreduce allgather along columns
C_BCAST_ROW_WEST = 16    # broadcast: row spread east->west (root on right)
C_BCAST_COL_NORTH = 17   # broadcast: column spread south->north
C_GATHER_COL_SOUTH = 18  # gather toward bottom row
C_GATHER_ROW_EAST = 19   # gather toward right column
C_SPARE_XY_0 = 20
C_SPARE_YX_0 = 21
C_SPARE_XY_1 = 22
C_SPARE_YX_1 = 23


# direction -> (dr, dc)
_DIRS = {
    "east": (0, 1),
    "west": (0, -1),
    "south": (1, 0),
    "north": (-1, 0),
}

# Catalog: color_id -> (name, kind, direction-or-mode)
# kind == "bus"     -> static directional dest table
# kind == "unicast" -> dst-dependent dimension-order fallback
CATALOG: dict[int, tuple[str, str, str]] = {
    C_UNICAST_XY: ("unicast_xy", "unicast", "xy"),
    C_UNICAST_YX: ("unicast_yx", "unicast", "yx"),
    C_BCAST_ROW_EAST: ("bcast_row_east", "bus", "east"),
    C_BCAST_COL_SOUTH: ("bcast_col_south", "bus", "south"),
    C_GATHER_COL_NORTH: ("gather_col_north", "bus", "north"),
    C_GATHER_ROW_WEST: ("gather_row_west", "bus", "west"),
    C_REDUCE_COL_NORTH: ("reduce_col_north", "bus", "north"),
    C_REDUCE_ROW_WEST: ("reduce_row_west", "bus", "west"),
    C_AG_ROW_EAST: ("allgather_row_east", "bus", "east"),
    C_AG_ROW_WEST: ("allgather_row_west", "bus", "west"),
    C_AG_COL_SOUTH: ("allgather_col_south", "bus", "south"),
    C_AG_COL_NORTH: ("allgather_col_north", "bus", "north"),
    C_AR_RS_ROW_EAST: ("allreduce_rs_row_east", "bus", "east"),
    C_AR_AG_ROW_WEST: ("allreduce_ag_row_west", "bus", "west"),
    C_AR_RS_COL_SOUTH: ("allreduce_rs_col_south", "bus", "south"),
    C_AR_AG_COL_NORTH: ("allreduce_ag_col_north", "bus", "north"),
    C_BCAST_ROW_WEST: ("bcast_row_west", "bus", "west"),
    C_BCAST_COL_NORTH: ("bcast_col_north", "bus", "north"),
    C_GATHER_COL_SOUTH: ("gather_col_south", "bus", "south"),
    C_GATHER_ROW_EAST: ("gather_row_east", "bus", "east"),
    C_SPARE_XY_0: ("spare_xy_0", "unicast", "xy"),
    C_SPARE_YX_0: ("spare_yx_0", "unicast", "yx"),
    C_SPARE_XY_1: ("spare_xy_1", "unicast", "xy"),
    C_SPARE_YX_1: ("spare_yx_1", "unicast", "yx"),
}


def _set_bus(plan: ColorPlan, color_id: int, direction: str) -> None:
    """Install a static one-hop directional dest table for every node."""
    rows, cols = plan.rows, plan.cols
    dr, dc = _DIRS[direction]
    for node in range(rows * cols):
        r, c = divmod(node, cols)
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            plan.set_dest(node, color_id, {nr * cols + nc})
        else:
            plan.set_dest(node, color_id, set())


def _ensure_color(plan: ColorPlan, color_id: int, name: str) -> None:
    while len(plan.colors) <= color_id:
        plan.colors.append(Color(color_id=len(plan.colors), task=len(plan.colors) * 4))
    plan.colors[color_id] = Color(color_id=color_id, task=color_id * 4, name=name)


def build_ideal_plan(rows: int, cols: int, num_colors: int = MAX_COLORS) -> ColorPlan:
    """Instantiate the full fixed catalog for a given mesh size.

    The catalog is identical regardless of ``rows``/``cols``; only the dest
    tables are materialised for the concrete mesh.
    """
    num_colors = min(num_colors, MAX_COLORS)
    plan = ColorPlan(num_colors=num_colors, rows=rows, cols=cols)
    for cid in range(num_colors):
        name, kind, spec = CATALOG[cid]
        if kind == "unicast":
            plan.set_unicast_mode(cid, spec)
        else:
            _set_bus(plan, cid, spec)
        _ensure_color(plan, cid, name)
    return plan


def build_baseline_plan(rows: int, cols: int) -> ColorPlan:
    """Single virtual network, dimension-order XY (baseline)."""
    plan = ColorPlan(num_colors=1, rows=rows, cols=cols)
    plan.set_unicast_mode(0, "xy")
    _ensure_color(plan, 0, "single_vn_xy")
    return plan

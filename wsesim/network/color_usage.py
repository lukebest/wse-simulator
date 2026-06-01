"""Color usage rules, scenarios, and compile-time allocation helpers.

On-wafer color semantics follow Cerebras SDK 2.10.0 fabric docs and patent
US10,515,303; see ``docs/vendor/cerebras-sdk-2.10.0/``.  The separate
``cerebras-cloud-sdk`` PyPI package is a REST inference client and does **not**
expose NoC color APIs.  Implementation: ``wsesim.network.color*``.

See ``docs/color_usage_guide.md`` for the full scenario-by-scenario guide.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from wsesim.network.color import ColorPlan
from wsesim.network.color_alloc import is_acyclic_route
from wsesim.network.color_catalog import (
    C_AG_COL_NORTH,
    C_AG_COL_SOUTH,
    C_AG_ROW_EAST,
    C_AG_ROW_WEST,
    C_AR_AG_COL_NORTH,
    C_AR_AG_ROW_WEST,
    C_AR_RS_COL_SOUTH,
    C_AR_RS_ROW_EAST,
    C_BCAST_COL_NORTH,
    C_BCAST_COL_SOUTH,
    C_BCAST_ROW_EAST,
    C_BCAST_ROW_WEST,
    C_GATHER_COL_NORTH,
    C_GATHER_COL_SOUTH,
    C_GATHER_ROW_EAST,
    C_GATHER_ROW_WEST,
    C_REDUCE_COL_NORTH,
    C_REDUCE_ROW_WEST,
    C_REDUCE_COL_SOUTH,
    C_REDUCE_ROW_EAST,
    C_UNICAST_XY,
    C_UNICAST_YX,
    CATALOG,
    MAX_COLORS,
    build_ideal_plan,
)


class ColorBudget(str, Enum):
    """How many distinct colors one collective invocation may use."""

    MINIMAL = "minimal"    # 1 color — entire collective on one VN (unicast)
    COMPACT = "compact"    # 2 colors — time-multiplex opposing directions
    PARALLEL = "parallel"  # 3–4 colors — max link parallelism (current ideal)


# Per-pattern color count targets (distinct IDs used in one collective).
BUDGET_COLOR_COUNT: dict[ColorBudget, dict[str, int]] = {
    ColorBudget.MINIMAL: {
        "broadcast": 1,
        "gather": 1,
        "reduce": 1,
        "allgather": 1,
        "allreduce": 1,
    },
    ColorBudget.COMPACT: {
        "broadcast": 2,
        "gather": 2,
        "reduce": 2,
        "allgather": 2,
        "allreduce": 2,
    },
    ColorBudget.PARALLEL: {
        "broadcast": 2,
        "gather": 2,
        "reduce": 2,
        "allgather": 4,
        "allreduce": 4,
    },
}

# Primary / secondary color assignment per budget (parallel uses pattern-specific buses).
BUDGET_COLORS: dict[ColorBudget, tuple[int, int]] = {
    ColorBudget.MINIMAL: (C_UNICAST_XY, C_UNICAST_XY),
    ColorBudget.COMPACT: (C_UNICAST_XY, C_UNICAST_YX),
    ColorBudget.PARALLEL: (C_UNICAST_XY, C_UNICAST_YX),  # overridden per generator
}

# Patent / product profiles: same catalog semantics, fewer active IDs.
COLOR_PROFILES: dict[int, tuple[int, ...]] = {
    8: (
        C_UNICAST_XY,
        C_UNICAST_YX,
        C_BCAST_ROW_EAST,
        C_BCAST_COL_SOUTH,
        C_GATHER_COL_NORTH,
        C_REDUCE_COL_NORTH,
        C_AG_ROW_EAST,
        C_AR_RS_ROW_EAST,
    ),
    16: tuple(range(16)),
    24: tuple(range(24)),
}

SDK_SCOPE_NOTE = (
    "Cerebras SDK 2.10.0 (docs/vendor/cerebras-sdk-2.10.0/) defines on-wafer "
    "colors via CSL @get_color / @set_color_config. cerebras-cloud-sdk is "
    "only the Cloud REST API and has no fabric APIs. wsesim models compile-time "
    "color routing per docs/color_mechanism_analysis.md (US10,515,303)."
)


@dataclass(frozen=True, slots=True)
class ColorRule:
    """One static routing rule in the catalog."""

    color_id: int
    name: str
    kind: str  # "bus" | "unicast"
    spec: str  # compass direction or "xy"/"yx"
    task_offset: int = 0  # instruction_addr = base + color_id * 4


@dataclass(frozen=True, slots=True)
class CollectiveScenario:
    """How a collective pattern should use colors on the mesh."""

    pattern: str
    colors: tuple[int, ...]
    route_shape: str
    ordering: str
    pipelining: str
    notes: str = ""


def catalog_rules(num_colors: int = MAX_COLORS) -> list[ColorRule]:
    n = min(num_colors, MAX_COLORS)
    return [
        ColorRule(cid, name, kind, spec, task_offset=cid * 4)
        for cid, (name, kind, spec) in sorted(CATALOG.items())
        if cid < n
    ]


def pick_tree_colors(root: int, rows: int, cols: int, *, gather: bool) -> tuple[int, int]:
    """Row + column bus colors for spanning-tree collectives from ``root``."""
    r, c = divmod(root, cols)
    if gather:
        if c == 0:
            row_c = C_GATHER_ROW_WEST
        elif c == cols - 1:
            row_c = C_GATHER_ROW_EAST
        else:
            row_c = C_GATHER_ROW_WEST
        if r == 0:
            col_c = C_GATHER_COL_NORTH
        elif r == rows - 1:
            col_c = C_GATHER_COL_SOUTH
        else:
            col_c = C_GATHER_COL_NORTH
    else:
        if c == 0:
            row_c = C_BCAST_ROW_EAST
        elif c == cols - 1:
            row_c = C_BCAST_ROW_WEST
        else:
            row_c = C_BCAST_ROW_EAST
        if r == 0:
            col_c = C_BCAST_COL_SOUTH
        elif r == rows - 1:
            col_c = C_BCAST_COL_NORTH
        else:
            col_c = C_BCAST_COL_SOUTH
    return row_c, col_c


def pick_reduce_colors(root: int, rows: int, cols: int) -> tuple[int, int]:
    """Dedicated reduce VNs (separate from gather even when shape matches)."""
    row_c, col_c = pick_tree_colors(root, rows, cols, gather=True)
    row_map = {
        C_GATHER_ROW_WEST: C_REDUCE_ROW_WEST,
        C_GATHER_ROW_EAST: C_REDUCE_ROW_EAST,
    }
    col_map = {
        C_GATHER_COL_NORTH: C_REDUCE_COL_NORTH,
        C_GATHER_COL_SOUTH: C_REDUCE_COL_SOUTH,
    }
    return row_map.get(row_c, C_REDUCE_ROW_WEST), col_map.get(col_c, C_REDUCE_COL_NORTH)


def pick_broadcast_colors(root: int, rows: int, cols: int) -> tuple[int, int]:
    return pick_tree_colors(root, rows, cols, gather=False)


COLLECTIVE_SCENARIOS: tuple[CollectiveScenario, ...] = (
    CollectiveScenario(
        pattern="broadcast",
        colors=(C_BCAST_ROW_EAST, C_BCAST_COL_SOUTH, C_BCAST_ROW_WEST, C_BCAST_COL_NORTH),
        route_shape="Spanning tree: row ripple then column ripple from root",
        ordering="One active source per (color, src, dst) stream; tree levels pipelined via delay_cycles",
        pipelining="Level d at delay=d; row phase then column phase",
        notes="Root at corner uses east+south; other corners use west/north variants (colors 16–17).",
    ),
    CollectiveScenario(
        pattern="gather",
        colors=(C_GATHER_COL_NORTH, C_GATHER_ROW_WEST, C_GATHER_COL_SOUTH, C_GATHER_ROW_EAST),
        route_shape="Reverse tree: columns converge to root row, then row converges to root",
        ordering="Per-color FIFO; concurrent columns use same col_color but disjoint links",
        pipelining="Column levels 0..rows-2, then row levels",
        notes="Separate VN from reduce so gather and reduce can overlap in fused kernels.",
    ),
    CollectiveScenario(
        pattern="reduce",
        colors=(C_REDUCE_COL_NORTH, C_REDUCE_ROW_WEST, C_REDUCE_COL_SOUTH, C_REDUCE_ROW_EAST),
        route_shape="Same geometry as gather; independent color IDs",
        ordering="Same as gather",
        pipelining="Same as gather",
        notes="Partial sums merge at each hop in CE; payload size unchanged in sim.",
    ),
    CollectiveScenario(
        pattern="allgather",
        colors=(C_AG_ROW_EAST, C_AG_ROW_WEST, C_AG_COL_SOUTH, C_AG_COL_NORTH),
        route_shape="2D ring: bidirectional row sweep then bidirectional column sweep",
        ordering="Ring steps time-phased; east/west and south/north are separate VNs",
        pipelining="cols-1 row steps, then rows-1 column steps",
        notes="Each step every PE sends one chunk to ring neighbor; 4 colors avoid link sharing.",
    ),
    CollectiveScenario(
        pattern="allreduce",
        colors=(
            C_AR_RS_ROW_EAST,
            C_AR_AG_ROW_WEST,
            C_AR_RS_COL_SOUTH,
            C_AR_AG_COL_NORTH,
        ),
        route_shape="Reduce-scatter then allgather: rows (RS east, AG west) then columns (RS south, AG north)",
        ordering="Four phases non-overlapping in time (t offsets); each phase one color",
        pipelining="Phase boundaries at t += max(1, cols-1) and t += max(1, rows-1)",
        notes="Mirrors 2D decomposed allreduce; each RS/AG pair reuses ring geometry with dedicated VNs.",
    ),
    CollectiveScenario(
        pattern="unicast",
        colors=(C_UNICAST_XY, C_UNICAST_YX),
        route_shape="Dimension-order unicast; dst-dependent static XY/YX step",
        ordering="Stripe concurrent flows across colors 0/1 by injection index",
        pipelining="None (point-to-point)",
        notes="Fallback for irregular or compile-time unknown dst; not minimal for collectives.",
    ),
)


def distinct_colors(traffic: list[dict]) -> int:
    """Number of distinct color IDs in one collective traffic batch."""
    return len({int(p.get("color", 0)) for p in traffic})


def budget_for(pattern: str, budget: ColorBudget) -> int:
    """Target distinct color count for one collective under a budget policy."""
    return BUDGET_COLOR_COUNT[budget].get(pattern.lower(), 1)


def primary_secondary(budget: ColorBudget) -> tuple[int, int]:
    """Return (primary, secondary) color IDs for compact/minimal ring scheduling."""
    return BUDGET_COLORS[budget]


def minimize_color_explanation() -> str:
    """Why minimal/compact work and when to use each budget."""
    return (
        "Directional bus colors encode ONE compass direction per node; a tree needs "
        "≥2 bus colors (row + column axes) unless unicast is used. "
        "MINIMAL assigns the whole collective to one unicast VN (color 0): phases "
        "and opposing ring directions are time-serialized via delay_cycles, so "
        "flows never overlap on the same color+link. "
        "COMPACT uses two unicast VNs (XY + YX): forward hops on color 0, "
        "backward hops on color 1, reusing the same pair across row/column phases "
        "because phases do not overlap in time. "
        "PARALLEL dedicates separate bus colors per direction for maximum "
        "concurrency (current default)."
    )


def scenario_for(pattern: str) -> CollectiveScenario | None:
    p = pattern.lower()
    for s in COLLECTIVE_SCENARIOS:
        if s.pattern == p:
            return s
    return None


def validate_plan(plan: ColorPlan) -> dict[str, list[int]]:
    """Return acyclicity report: color_id -> [] if OK, else [nodes in cycle hint]."""
    bad: dict[str, list[int]] = {}
    for cid in range(plan.num_colors):
        if cid in plan.unicast_modes:
            continue
        if not is_acyclic_route(plan.dest, cid, plan.num_nodes):
            bad[str(cid)] = []
    return bad


def build_profile_plan(rows: int, cols: int, profile: int = 24) -> ColorPlan:
    """Build ideal plan restricted to a color profile (8 / 16 / 24)."""
    if profile not in COLOR_PROFILES:
        raise ValueError(f"Unsupported profile {profile}; choose from {sorted(COLOR_PROFILES)}")
    active = COLOR_PROFILES[profile]
    full = build_ideal_plan(rows, cols, max(active) + 1)
    full.num_colors = len(active)
    return full


def task_id_for_color(color_id: int) -> int:
    """Wavelet task selector: instruction_addr = base + color × 4."""
    return color_id * 4


def color_field_summary() -> str:
    """One-paragraph summary for docs / CLI."""
    return (
        f"Fixed catalog of up to {MAX_COLORS} virtual networks. "
        "Each color is a static directional bus or XY/YX unicast mode. "
        "Collectives pick colors by pattern (see COLLECTIVE_SCENARIOS). "
        + SDK_SCOPE_NOTE
    )

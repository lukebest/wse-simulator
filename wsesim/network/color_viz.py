"""Schedule builder for Color mechanism visualization (mirrors docs/color_mesh_viz.html)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from wsesim.network.color import ColorPlan
from wsesim.network.color_catalog import CATALOG, build_ideal_plan
from wsesim.network.collective import generate_baseline_collective, generate_ideal_collective
from wsesim.network.color_usage import ColorBudget, distinct_colors


@dataclass(slots=True)
class VizHop:
    src: int
    dst: int

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst}


@dataclass(slots=True)
class VizFlow:
    id: str
    src: int
    dst: int
    color_id: int
    color_name: str
    flits: int
    start: int
    period: int
    payload: str
    stage: str
    path: list[VizHop] = field(default_factory=list)
    vn_stream: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = [h.to_dict() for h in self.path]
        return d


@dataclass(slots=True)
class VizSchedule:
    rows: int
    cols: int
    pattern: str
    budget: str
    flits: int
    duration: int
    peak: int
    distinct_colors: int
    flows: list[VizFlow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "pattern": self.pattern,
            "budget": self.budget,
            "flits": self.flits,
            "duration": self.duration,
            "peak": self.peak,
            "distinct_colors": self.distinct_colors,
            "flows": [f.to_dict() for f in self.flows],
        }


def expand_path(plan: ColorPlan, src: int, dst: int, color_id: int) -> list[tuple[int, int]]:
    """Expand (src, dst, color) to hop list using static bus or unicast routing."""
    if src == dst:
        return []
    hops: list[tuple[int, int]] = []
    current = src
    guard = 0
    while current != dst and guard <= plan.num_nodes + 2:
        nxt_set = plan.next_hops(current, color_id, dst)
        if not nxt_set:
            break
        nxt = min(nxt_set)
        hops.append((current, nxt))
        current = nxt
        guard += 1
    return hops


def _color_name(color_id: int) -> str:
    if color_id in CATALOG:
        return CATALOG[color_id][0]
    return f"color_{color_id}"


def _flow_duration(flow: VizFlow) -> int:
    return (flow.start or 0) + (flow.flits - 1) * flow.period + max(1, len(flow.path))


def _peak_edge_occupancy(flows: list[VizFlow]) -> tuple[int, int]:
    if not flows:
        return 0, 0
    duration = max(_flow_duration(f) for f in flows)
    peak = 0
    for cycle in range(duration + 1):
        counts: dict[tuple[int, int], int] = {}
        for f in flows:
            for flit in range(f.flits):
                off = cycle - (f.start + flit * f.period)
                if 0 <= off < len(f.path):
                    edge = f.path[off]
                    key = (edge.src, edge.dst)
                    counts[key] = counts.get(key, 0) + 1
        if counts:
            peak = max(peak, max(counts.values()))
    return duration, peak


def traffic_to_flows(
    traffic: list[dict],
    plan: ColorPlan,
    *,
    flits: int = 1,
) -> list[VizFlow]:
    """Convert collective traffic packets to viz flows with expanded paths."""
    flows: list[VizFlow] = []
    for idx, pkt in enumerate(traffic):
        src = int(pkt["src_core"])
        dst = int(pkt["dst_core"])
        color_id = int(pkt.get("color", 0))
        start = int(pkt.get("delay_cycles", 0))
        payload = str(pkt.get("payload", "data"))
        raw_hops = expand_path(plan, src, dst, color_id)
        path = [VizHop(s, d) for s, d in raw_hops]
        vn = f"c{color_id}:{src}->{dst}"
        flows.append(
            VizFlow(
                id=f"F{idx}",
                src=src,
                dst=dst,
                color_id=color_id,
                color_name=_color_name(color_id),
                flits=max(1, flits),
                start=start,
                period=1,
                payload=payload,
                stage=payload,
                path=path,
                vn_stream=vn,
            )
        )
    return flows


def build_viz_schedule(
    rows: int,
    cols: int,
    pattern: str,
    *,
    budget: ColorBudget | str = ColorBudget.PARALLEL,
    root: int = 0,
    flits: int = 1,
    chunk_bytes: int = 128,
) -> VizSchedule:
    """Build a flit-pipeline schedule for visualization."""
    if isinstance(budget, str):
        if budget == "xy_baseline":
            return build_baseline_schedule(
                rows, cols, pattern, flits=flits, chunk_bytes=chunk_bytes
            )
        budget = ColorBudget(budget)

    if budget == ColorBudget.MINIMAL:
        traffic = generate_ideal_collective(
            pattern, rows, cols, chunk_bytes, root=root, color_budget=ColorBudget.MINIMAL
        )
        budget_label = "minimal"
    elif budget == ColorBudget.COMPACT:
        traffic = generate_ideal_collective(
            pattern, rows, cols, chunk_bytes, root=root, color_budget=ColorBudget.COMPACT
        )
        budget_label = "compact"
    else:
        traffic = generate_ideal_collective(
            pattern, rows, cols, chunk_bytes, root=root, color_budget=ColorBudget.PARALLEL
        )
        budget_label = "parallel"

    plan = build_ideal_plan(rows, cols)
    flows = traffic_to_flows(traffic, plan, flits=flits)
    duration, peak = _peak_edge_occupancy(flows)
    return VizSchedule(
        rows=rows,
        cols=cols,
        pattern=pattern,
        budget=budget_label,
        flits=flits,
        duration=duration,
        peak=peak,
        distinct_colors=distinct_colors(traffic),
        flows=flows,
    )


def build_baseline_schedule(
    rows: int,
    cols: int,
    pattern: str,
    *,
    flits: int = 1,
    chunk_bytes: int = 128,
) -> VizSchedule:
    plan = build_ideal_plan(rows, cols)
    traffic = generate_baseline_collective(pattern, rows, cols, chunk_bytes)
    flows = traffic_to_flows(traffic, plan, flits=flits)
    duration, peak = _peak_edge_occupancy(flows)
    return VizSchedule(
        rows=rows,
        cols=cols,
        pattern=pattern,
        budget="xy_baseline",
        flits=flits,
        duration=duration,
        peak=peak,
        distinct_colors=distinct_colors(traffic),
        flows=flows,
    )


def summarize_budgets(
    rows: int,
    cols: int,
    pattern: str,
    *,
    flits: int = 1,
    chunk_bytes: int = 128,
    root: int = 0,
) -> list[dict]:
    """Compare xy baseline + three color budgets."""
    rows_out: list[dict] = []
    base = build_baseline_schedule(rows, cols, pattern, flits=flits, chunk_bytes=chunk_bytes)
    rows_out.append(
        {
            "scheme": "xy_baseline",
            "distinct_colors": base.distinct_colors,
            "duration": base.duration,
            "peak": base.peak,
            "flows": len(base.flows),
        }
    )
    for budget in (ColorBudget.MINIMAL, ColorBudget.COMPACT, ColorBudget.PARALLEL):
        sched = build_viz_schedule(
            rows, cols, pattern, budget=budget, root=root, flits=flits, chunk_bytes=chunk_bytes
        )
        rows_out.append(
            {
                "scheme": f"color_{budget.value}",
                "distinct_colors": sched.distinct_colors,
                "duration": sched.duration,
                "peak": sched.peak,
                "flows": len(sched.flows),
            }
        )
    return rows_out

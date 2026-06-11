#!/usr/bin/env python3
"""Analyze all six collectives and refresh conflict_free_collectives_report.html."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsesim.network.collective_patterns import analyze_all_collectives  # noqa: E402

MESHES = [(4, 4), (8, 8), (12, 16)]
OUT_JSON = ROOT / "outputs/collectives_report/results.json"
OUT_HTML = ROOT / "docs/conflict_free_collectives_report.html"

PATTERN_ORDER = (
    "broadcast",
    "reduce",
    "allreduce",
    "allgather",
    "gather",
    "alltoall",
)
PATTERN_LABELS = {
    "broadcast": "Broadcast",
    "reduce": "Reduce",
    "allreduce": "AllReduce",
    "allgather": "AllGather",
    "gather": "Gather",
    "alltoall": "AllToAll",
    "alltoall_twophase": "AllToAll (2-phase stall=0)",
}


def run_analysis() -> list[dict]:
    return analyze_all_collectives(MESHES)


def _mesh_label(rows: int, cols: int) -> str:
    return f"{rows}×{cols}"


def _ok_cell(ok: bool) -> str:
    return "<td class='good'>✓</td>" if ok else "<td class='warn'>✗</td>"


def _stall_cell(stall: int) -> str:
    cls = "good" if stall == 0 else "warn"
    return f"<td class='{cls}'>{stall}</td>"


def _master_table_rows(results: list[dict]) -> str:
    lines: list[str] = []
    for r in results:
        mesh = _mesh_label(r["rows"], r["cols"])
        pat = PATTERN_LABELS.get(r["pattern"], r["pattern"])
        tight = (
            "Z=L*"
            if r["router_table_oneshot"] == r["router_table_periodic"]
            else f"P&lt;Z ({r['router_table_periodic']}&lt;{r['router_table_oneshot']})"
        )
        lines.append(
            "<tr>"
            f"<td>{pat}</td>"
            f"<td>{mesh}</td>"
            f"<td>{r['z_lower_bound']}</td>"
            f"<td class='good'>{r['z_scheduled']}</td>"
            f"<td>{r['l_star']}</td>"
            f"<td>{r['router_table_oneshot']}</td>"
            f"<td>{r['router_max_active_span']}</td>"
            f"<td>{tight}</td>"
            f"{_stall_cell(r['stall'])}"
            f"{_ok_cell(r['ok'])}"
            "</tr>"
        )
    return "\n".join(lines)


def _router_summary_rows(results: list[dict]) -> str:
    """One row per mesh: min periodic vs max one-shot across patterns."""
    lines: list[str] = []
    for rows, cols in MESHES:
        subset = [r for r in results if r["rows"] == rows and r["cols"] == cols]
        p_min = min(r["router_table_periodic"] for r in subset)
        z_max = max(r["router_table_oneshot"] for r in subset)
        z_tight = [r for r in subset if r["router_table_oneshot"] == r["l_star"]]
        names = ", ".join(PATTERN_LABELS.get(r["pattern"], r["pattern"]) for r in z_tight)
        lines.append(
            "<tr>"
            f"<td>{_mesh_label(rows, cols)}</td>"
            f"<td class='good'>{p_min}</td>"
            f"<td>{z_max}</td>"
            f"<td>{names or '—'}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def inject_results_into_html(html: str, results: list[dict]) -> str:
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    master = _master_table_rows(results)
    router = _router_summary_rows(results)
    html = re.sub(
        r"(<tbody id=\"master-body\">)(.*?)(</tbody>)",
        rf"\1\n{master}\n\3",
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r"(<tbody id=\"router-body\">)(.*?)(</tbody>)",
        rf"\1\n{router}\n\3",
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r"(<script id=\"bench-data\" type=\"application/json\">)(.*?)(</script>)",
        rf"\1\n{payload}\n\3",
        html,
        count=1,
        flags=re.S,
    )
    return html


def main() -> None:
    results = run_analysis()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    html = OUT_HTML.read_text(encoding="utf-8")
    OUT_HTML.write_text(inject_results_into_html(html, results), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Updated {OUT_HTML}")
    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"WARNING: {len(failed)} schedules failed verification")
    else:
        print(f"All {len(results)} analyses ok=True (edge conflict-free at assigned slots)")


if __name__ == "__main__":
    main()

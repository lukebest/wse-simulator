#!/usr/bin/env python3
"""Generate fault-tolerance degradation report for mesh collectives."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsesim.network.collective_faults import (  # noqa: E402
    FAULT_TYPES,
    PATTERNS,
    REGIONS,
    analyze_fault_matrix,
)

MESHES = [(4, 4), (8, 8), (12, 16)]
OUT_JSON = ROOT / "outputs/fault_tolerance/results.json"
OUT_HTML = ROOT / "docs/fault_tolerant_collectives_report.html"

PATTERN_LABELS = {
    "broadcast": "Broadcast",
    "reduce": "Reduce",
    "allreduce": "AllReduce",
    "allgather": "AllGather",
    "gather": "Gather",
    "alltoall": "AllToAll",
}

FAULT_LABELS = {
    "pe_point": "PE 坏点",
    "link_point": "链路坏",
    "pe_block": "PE 坏块 (2×2)",
}

REGION_LABELS = {
    "corner": "角",
    "edge": "边",
    "center": "中心",
}


def run_analysis() -> list[dict]:
    return analyze_fault_matrix(MESHES)


def _ratio_cell(ratio: float | None) -> str:
    if ratio is None:
        return "<td class='warn'>—</td>"
    cls = "good" if ratio <= 1.05 else ("warn" if ratio <= 1.5 else "bad")
    return f"<td class='{cls}'>{ratio:.3f}×</td>"


def _master_table_rows(results: list[dict]) -> str:
    lines: list[str] = []
    for r in results:
        pat = PATTERN_LABELS.get(r["pattern"], r["pattern"])
        ft = FAULT_LABELS.get(r["fault_type"], r["fault_type"])
        reg = REGION_LABELS.get(r["region"], r["region"])
        note = r.get("note") or "—"
        lines.append(
            "<tr>"
            f"<td>{pat}</td>"
            f"<td>{r['mesh']}</td>"
            f"<td>{ft}</td>"
            f"<td>{reg}</td>"
            f"<td>{r['z_healthy_ref']}</td>"
            f"<td>{r['z_healthy_scheduled']}</td>"
            f"<td>{r['z_faulty']}</td>"
            f"{_ratio_cell(r.get('ratio'))}"
            f"<td>{r['stall_faulty']}</td>"
            f"<td>{r['peak_faulty']}</td>"
            f"<td>{note}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def _summary_by_mesh(results: list[dict]) -> str:
    lines: list[str] = []
    for rows, cols in MESHES:
        mesh = f"{rows}×{cols}"
        subset = [r for r in results if r["rows"] == rows and r["cols"] == cols]
        if not subset:
            continue
        max_r = max(r for r in (x.get("ratio") or 0 for x in subset))
        worst = max(subset, key=lambda x: x.get("ratio") or 0)
        ham = [r for r in subset if r.get("hamilton_degraded")]
        lines.append(
            "<tr>"
            f"<td>{mesh}</td>"
            f"<td>{len(subset)}</td>"
            f"<td class='bad'>{max_r:.3f}×</td>"
            f"<td>{PATTERN_LABELS.get(worst['pattern'], worst['pattern'])}</td>"
            f"<td>{FAULT_LABELS.get(worst['fault_type'], worst['fault_type'])}</td>"
            f"<td>{REGION_LABELS.get(worst['region'], worst['region'])}</td>"
            f"<td>{len(ham)}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def inject_results_into_html(html: str, results: list[dict]) -> str:
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    master = _master_table_rows(results)
    summary = _summary_by_mesh(results)
    html = re.sub(
        r'(<tbody id="master-body">)(.*?)(</tbody>)',
        rf"\1\n{master}\n\3",
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r'(<tbody id="summary-body">)(.*?)(</tbody>)',
        rf"\1\n{summary}\n\3",
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r'(<script id="bench-data" type="application/json">)(.*?)(</script>)',
        rf"\1\n{payload}\n\3",
        html,
        count=1,
        flags=re.S,
    )
    return html


def main() -> None:
    results = run_analysis()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    html = OUT_HTML.read_text(encoding="utf-8")
    OUT_HTML.write_text(inject_results_into_html(html, results), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Updated {OUT_HTML}")
    print(f"Analyzed {len(results)} fault scenarios")


if __name__ == "__main__":
    main()

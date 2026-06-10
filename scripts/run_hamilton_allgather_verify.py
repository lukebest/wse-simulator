#!/usr/bin/env python3
"""Run Hamiltonian vs dimwise allgather benchmark and refresh HTML results."""

from __future__ import annotations

import json
import re
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsesim.network.collective_patterns import compare_allgather_patterns  # noqa: E402


def run_benchmark() -> list[dict]:
    meshes = [(4, 4), (8, 8), (12, 16)]
    return [compare_allgather_patterns(rows, cols, 1) for rows, cols in meshes]
OUT_JSON = ROOT / "outputs/hamilton_allgather/results.json"
OUT_HTML = ROOT / "docs/hamilton_ring_allgather.html"


def _fmt_speedup(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}×"


def _table_rows(results: list[dict]) -> str:
    lines: list[str] = []
    for r in results:
        mesh = f"{r['rows']}×{r['cols']}"
        n = r["node_count"]
        t_star = r["t_star"]
        h = r["hamilton"]
        d = r["dimwise"]
        lines.append(
            "<tr>"
            f"<td>{mesh}</td>"
            f"<td>{n}</td>"
            f"<td>⌈(N−1)/2⌉ = {t_star}</td>"
            f"<td class='good'>{h['makespan']}</td>"
            f"<td class='good'>{h['stall']}</td>"
            f"<td>{d['makespan']}</td>"
            f"<td>{d['stall']}</td>"
            f"<td>{d['l_star']}</td>"
            f"<td class='good'>{_fmt_speedup(r['speedup_vs_dimwise'])}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def inject_results_into_html(html: str, results: list[dict]) -> str:
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    table = _table_rows(results)
    html = re.sub(
        r"(<tbody id=\"bench-body\">)(.*?)(</tbody>)",
        rf"\1\n{table}\n\3",
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
    results = run_benchmark()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    html = OUT_HTML.read_text(encoding="utf-8")
    OUT_HTML.write_text(inject_results_into_html(html, results), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Updated {OUT_HTML}")


if __name__ == "__main__":
    main()

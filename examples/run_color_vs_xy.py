#!/usr/bin/env python3
"""Run color vs XY baseline study on 4x4 and 8x8 meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

from wsesim.network.color_sim import run_full_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Color NoC vs single-VN XY baseline")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/color_vs_xy"),
        help="Output directory",
    )
    parser.add_argument("--msg-bytes", type=int, default=128)
    args = parser.parse_args()

    meshes = [(4, 4), (8, 8)]
    for rows, cols in meshes:
        out = args.output / f"{rows}x{cols}"
        run_full_study([(rows, cols)], out, msg_bytes=args.msg_bytes)
        print(f"Wrote {out / 'results.csv'}")

    combined = args.output / "results.csv"
    rows_all = []
    for rows, cols in meshes:
        p = args.output / f"{rows}x{cols}" / "results.csv"
        if p.exists():
            rows_all.append(p.read_text(encoding="utf-8"))
    if rows_all:
        header = rows_all[0].splitlines()[0]
        body = []
        for chunk in rows_all:
            lines = chunk.splitlines()
            body.extend(lines[1:])
        combined.parent.mkdir(parents=True, exist_ok=True)
        combined.write_text(header + "\n" + "\n".join(body) + "\n", encoding="utf-8")
        print(f"Combined results: {combined}")


if __name__ == "__main__":
    main()

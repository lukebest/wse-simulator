#!/usr/bin/env -S .venv/bin/python
"""Run color NoC vs single-VN XY baseline study."""

from __future__ import annotations

import argparse
from pathlib import Path

from wsesim.network.color_sim import run_full_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Color NoC vs XY baseline simulation")
    parser.add_argument("--output", type=Path, default=Path("outputs/color_vs_xy"))
    parser.add_argument("--colors", type=int, default=16)
    parser.add_argument("--msg-bytes", type=int, default=128)
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=["4x4", "8x8", "16x16"],
        help="Mesh sizes as RxC",
    )
    args = parser.parse_args()

    mesh_sizes = []
    for m in args.meshes:
        r, c = m.split("x")
        mesh_sizes.append((int(r), int(c)))

    for rows, cols in mesh_sizes:
        label = f"{rows}x{cols}"
        out = args.output / label
        run_full_study(
            [(rows, cols)], out, num_colors=args.colors, msg_bytes=args.msg_bytes, write_trace=True
        )
        print(f"Completed {label} -> {out / 'results.csv'}")


if __name__ == "__main__":
    main()

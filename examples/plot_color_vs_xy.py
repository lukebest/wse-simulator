#!/usr/bin/env python3
"""Plot color NoC vs XY baseline makespan from results.csv files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def plot_makespan_bars(rows: list[dict], output: Path, title: str) -> None:
    algorithms = sorted({r["algorithm"] for r in rows})
    color_ms = []
    xy_ms = []
    speedups = []
    for algo in algorithms:
        c = next(r for r in rows if r["algorithm"] == algo and r["topology"] == "color_mesh2d")
        x = next(r for r in rows if r["algorithm"] == algo and r["topology"] == "mesh2d_xy")
        color_ms.append(int(c["makespan_cycles"]))
        xy_ms.append(int(x["makespan_cycles"]))
        speedups.append(int(x["makespan_cycles"]) / max(int(c["makespan_cycles"]), 1))

    x_pos = range(len(algorithms))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x_pos], color_ms, width, label="Color NoC", color="#2563eb")
    ax.bar([i + width / 2 for i in x_pos], xy_ms, width, label="XY baseline", color="#94a3b8")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(algorithms, rotation=30, ha="right")
    ax.set_ylabel("Makespan (cycles)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_speedup(rows: list[dict], output: Path, title: str) -> None:
    algorithms = sorted({r["algorithm"] for r in rows})
    speedups = []
    for algo in algorithms:
        c = next(r for r in rows if r["algorithm"] == algo and r["topology"] == "color_mesh2d")
        x = next(r for r in rows if r["algorithm"] == algo and r["topology"] == "mesh2d_xy")
        speedups.append(int(x["makespan_cycles"]) / max(int(c["makespan_cycles"]), 1))

    x_pos = range(len(algorithms))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(x_pos), speedups, color="#16a34a")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Speedup (XY / Color)")
    ax.set_title(title)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(algorithms, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/color_vs_xy"))
    parser.add_argument("--output", type=Path, default=Path("outputs/color_vs_xy/plots"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for mesh_dir in sorted(args.input.iterdir()):
        csv_path = mesh_dir / "results.csv"
        if not csv_path.exists():
            continue
        rows = load_results(csv_path)
        mesh = rows[0]["mesh"] if rows else mesh_dir.name
        plot_makespan_bars(
            rows,
            args.output / f"makespan_{mesh}.png",
            f"Color NoC vs XY — {mesh}",
        )
        plot_speedup(
            rows,
            args.output / f"speedup_{mesh}.png",
            f"XY/Color speedup — {mesh}",
        )
        print(f"Plotted {mesh}")


if __name__ == "__main__":
    main()

"""Plotting helpers for DSE CSV exports."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def plot_pareto(
    trials_csv: Path,
    pareto_csv: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    trials = _read_rows(trials_csv)
    pareto = _read_rows(pareto_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(
        _as_float(trials, "total_latency_cycles"),
        _as_float(trials, "network_throughput"),
        alpha=0.5,
        s=30,
        label="All Trials",
    )
    ax1.scatter(
        _as_float(pareto, "total_latency_cycles"),
        _as_float(pareto, "network_throughput"),
        alpha=0.9,
        s=50,
        marker="x",
        label="Pareto Front",
    )
    ax1.set_xlabel("Latency (cycles)")
    ax1.set_ylabel("Throughput (flits/cycle)")
    ax1.set_title("DSE: Latency vs Throughput")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    out_latency_thr = output_dir / "pareto_latency_vs_throughput.png"
    fig1.tight_layout()
    fig1.savefig(out_latency_thr, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    trial_congestion = [
        float(row["vc_wait_cycles"]) + float(row["buffer_wait_cycles"]) + float(row["link_wait_cycles"])
        for row in trials
    ]
    pareto_congestion = [
        float(row["vc_wait_cycles"]) + float(row["buffer_wait_cycles"]) + float(row["link_wait_cycles"])
        for row in pareto
    ]
    ax2.scatter(
        _as_float(trials, "total_latency_cycles"),
        trial_congestion,
        alpha=0.5,
        s=30,
        label="All Trials",
    )
    ax2.scatter(
        _as_float(pareto, "total_latency_cycles"),
        pareto_congestion,
        alpha=0.9,
        s=50,
        marker="x",
        label="Pareto Front",
    )
    ax2.set_xlabel("Latency (cycles)")
    ax2.set_ylabel("Congestion (VC+Buffer+Link wait cycles)")
    ax2.set_title("DSE: Latency vs Congestion")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    out_latency_cong = output_dir / "pareto_latency_vs_congestion.png"
    fig2.tight_layout()
    fig2.savefig(out_latency_cong, dpi=150)
    plt.close(fig2)

    return out_latency_thr, out_latency_cong

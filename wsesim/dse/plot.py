"""Plotting helpers for DSE CSV exports."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wsesim.dse.pipeline_analysis import PipelineBreakdown


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


def plot_pipeline_gantt(breakdowns: list[PipelineBreakdown], output_path: Path) -> Path:
    if not breakdowns:
        raise ValueError("No pipeline breakdowns provided for Gantt plot.")

    color_map = {
        "compute": "#4e79a7",
        "memory": "#f28e2b",
        "network": "#59a14f",
        "io": "#e15759",
        "allreduce": "#b07aa1",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(5, len(breakdowns) * 0.7)))
    y_ticks: list[float] = []
    y_labels: list[str] = []

    for idx, breakdown in enumerate(breakdowns):
        y = float(idx)
        y_ticks.append(y)
        y_labels.append(breakdown.config_label)
        for stage in breakdown.stages:
            if stage.duration_cycles <= 0:
                continue
            ax.barh(
                y,
                stage.duration_cycles,
                left=stage.start_cycle,
                height=0.55,
                color=color_map.get(stage.category, "#9c9c9c"),
                edgecolor="white",
                linewidth=0.4,
            )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Cycles")
    ax.set_title("DeepSeek FFN Pipeline Timeline by Partition Strategy")
    ax.grid(True, axis="x", alpha=0.2)
    ax.invert_yaxis()

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[name])
        for name in ("compute", "memory", "network", "io", "allreduce")
    ]
    ax.legend(handles, ["compute", "memory", "network", "io", "allreduce"], loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_latency_breakdown(breakdowns: list[PipelineBreakdown], output_path: Path) -> Path:
    if not breakdowns:
        raise ValueError("No pipeline breakdowns provided for breakdown plot.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [b.config_label for b in breakdowns]
    category_order = ["compute", "memory", "network", "io", "allreduce"]
    color_map = {
        "compute": "#4e79a7",
        "memory": "#f28e2b",
        "network": "#59a14f",
        "io": "#e15759",
        "allreduce": "#b07aa1",
    }

    agg = [b.category_cycles() for b in breakdowns]
    x = list(range(len(labels)))
    bottoms = [0] * len(labels)

    fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.7)))
    for category in category_order:
        vals = [row.get(category, 0) for row in agg]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            color=color_map[category],
            label=category,
            width=0.65,
        )
        bottoms = [bottoms[i] + vals[i] for i in range(len(vals))]

    for idx, row in enumerate(agg):
        dominant = max(category_order, key=lambda cat: row.get(cat, 0))
        total = sum(row.values())
        ax.text(idx, total * 1.01, dominant, ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Cycles")
    ax.set_title("Latency Breakdown by Pipeline Category")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_bandwidth_utilization(breakdowns: list[PipelineBreakdown], output_path: Path) -> Path:
    if not breakdowns:
        raise ValueError("No pipeline breakdowns provided for bandwidth plot.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [b.config_label for b in breakdowns]
    mem_util = [b.peak_mem_bw_utilization * 100.0 for b in breakdowns]
    compute_util = []
    for breakdown in breakdowns:
        category = breakdown.category_cycles()
        compute = category.get("compute", 0)
        compute_util.append((compute / max(1, breakdown.total_cycles)) * 100.0)

    x = list(range(len(labels)))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(12, max(5, len(labels) * 0.7)))
    ax2 = ax1.twinx()

    ax1.bar([i - width / 2 for i in x], mem_util, width=width, label="Memory BW Utilization (%)")
    ax2.bar([i + width / 2 for i in x], compute_util, width=width, label="Compute Utilization (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Memory BW Utilization (%)")
    ax2.set_ylabel("Compute Utilization (%)")
    ax1.set_ylim(0, max(105, max(mem_util) + 10))
    ax2.set_ylim(0, max(105, max(compute_util) + 10))
    ax1.axhline(100.0, linestyle="--", linewidth=1.0)
    ax1.set_title("Memory Bandwidth vs Compute Utilization")
    ax1.grid(True, axis="y", alpha=0.2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

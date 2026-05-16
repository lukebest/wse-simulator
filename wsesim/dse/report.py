"""DSE report helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wsesim.core.config import WSEConfig
from wsesim.dse.engine import DSETrial


def summarize_best(history: list[tuple[WSEConfig, float]]) -> dict[str, object]:
    if not history:
        return {"best_score": None, "best_config": None, "trials": 0}
    best_cfg, best_score = max(history, key=lambda item: item[1])
    return {"best_score": best_score, "best_config": asdict(best_cfg), "trials": len(history)}


def summarize_best_trial(trials: list[DSETrial]) -> dict[str, Any]:
    if not trials:
        return {"best_score": None, "best_config": None, "best_result": None, "trials": 0}
    best = max(trials, key=lambda trial: trial.score)
    return {
        "best_score": best.score,
        "best_config": asdict(best.config),
        "best_result": asdict(best.result),
        "trials": len(trials),
    }


def pareto_front(
    trials: list[DSETrial],
    minimize_metrics: tuple[str, ...] = (
        "total_latency_cycles",
        "vc_wait_cycles",
        "buffer_wait_cycles",
        "link_wait_cycles",
        "gateway_noc_hops",
        "gateway_peak_load",
    ),
    maximize_metrics: tuple[str, ...] = ("network_throughput",),
) -> list[DSETrial]:
    front: list[DSETrial] = []
    for candidate in trials:
        dominated = False
        for other in trials:
            if other is candidate:
                continue
            if _dominates(other, candidate, minimize_metrics, maximize_metrics):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _dominates(
    lhs: DSETrial,
    rhs: DSETrial,
    minimize_metrics: tuple[str, ...],
    maximize_metrics: tuple[str, ...],
) -> bool:
    lhs_values = lhs.result
    rhs_values = rhs.result
    better_or_equal_all = True
    strictly_better = False

    for metric in minimize_metrics:
        lhs_metric = float(getattr(lhs_values, metric, 0.0))
        rhs_metric = float(getattr(rhs_values, metric, 0.0))
        if lhs_metric > rhs_metric:
            better_or_equal_all = False
            break
        if lhs_metric < rhs_metric:
            strictly_better = True

    if better_or_equal_all:
        for metric in maximize_metrics:
            lhs_metric = float(getattr(lhs_values, metric, 0.0))
            rhs_metric = float(getattr(rhs_values, metric, 0.0))
            if lhs_metric < rhs_metric:
                better_or_equal_all = False
                break
            if lhs_metric > rhs_metric:
                strictly_better = True

    return better_or_equal_all and strictly_better


def export_trials_json(trials: list[DSETrial], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "score": trial.score,
            "config": asdict(trial.config),
            "result": asdict(trial.result),
        }
        for trial in trials
    ]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def export_trials_csv(trials: list[DSETrial], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_idx",
        "score",
        "total_latency_cycles",
        "vc_wait_cycles",
        "buffer_wait_cycles",
        "link_wait_cycles",
        "pipeline_cycles",
        "gateway_noc_hops",
        "gateway_peak_load",
        "network_throughput",
        "network_saturation",
        "pe_width",
        "noc_num_vcs",
        "noc_buffer_depth",
        "gateways_per_reticle",
        "gateway_policy",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, trial in enumerate(trials):
            writer.writerow(
                {
                    "trial_idx": idx,
                    "score": trial.score,
                    "total_latency_cycles": trial.result.total_latency_cycles,
                    "vc_wait_cycles": trial.result.vc_wait_cycles,
                    "buffer_wait_cycles": trial.result.buffer_wait_cycles,
                    "link_wait_cycles": trial.result.link_wait_cycles,
                    "pipeline_cycles": trial.result.pipeline_cycles,
                    "gateway_noc_hops": trial.result.gateway_noc_hops,
                    "gateway_peak_load": trial.result.gateway_peak_load,
                    "network_throughput": trial.result.network_throughput,
                    "network_saturation": trial.result.network_saturation,
                    "pe_width": trial.config.compute.pe_width,
                    "noc_num_vcs": trial.config.network.noc.num_vcs,
                    "noc_buffer_depth": trial.config.network.noc.buffer_depth,
                    "gateways_per_reticle": trial.config.network.gateways_per_reticle,
                    "gateway_policy": trial.config.network.gateway_policy,
                }
            )
    return output


def export_pareto_csv(front: list[DSETrial], output_path: str | Path) -> Path:
    return export_trials_csv(front, output_path)

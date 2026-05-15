"""Simulation results and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimResult:
    total_latency_cycles: int = 0
    compute_cycles: int = 0
    network_cycles: int = 0
    memory_stall_cycles: int = 0
    network_avg_latency: float = 0.0
    network_max_latency: float = 0.0
    network_throughput: float = 0.0
    network_saturation: float = 0.0
    idle_ratio: float = 0.0
    per_core_utilization: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def dse_score(self, weights: dict[str, float] | None = None) -> float:
        if not weights:
            return -float(self.total_latency_cycles)

        score = 0.0
        for metric_name, metric_weight in weights.items():
            if hasattr(self, metric_name):
                score += metric_weight * float(getattr(self, metric_name))
        return score

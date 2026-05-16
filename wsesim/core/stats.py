"""Simulation results and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SimResult:
    total_latency_cycles: int = 0
    compute_cycles: int = 0
    network_cycles: int = 0
    io_injection_cycles: int = 0
    memory_stall_cycles: int = 0
    allreduce_cycles: int = 0
    network_avg_latency: float = 0.0
    network_max_latency: float = 0.0
    network_throughput: float = 0.0
    network_saturation: float = 0.0
    vc_wait_cycles: int = 0
    buffer_wait_cycles: int = 0
    link_wait_cycles: int = 0
    pipeline_cycles: int = 0
    gateway_noc_hops: int = 0
    gateway_peak_load: int = 0
    idle_ratio: float = 0.0
    per_core_utilization: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def update_from_network_stats(self, stats: Any, sim_time_cycles: int | None = None) -> None:
        self.network_avg_latency = float(stats.avg_latency())
        self.network_max_latency = float(getattr(stats, "max_packet_latency", 0.0))
        self.vc_wait_cycles = int(getattr(stats, "vc_wait_cycles", 0))
        self.buffer_wait_cycles = int(getattr(stats, "buffer_wait_cycles", 0))
        self.link_wait_cycles = int(getattr(stats, "link_wait_cycles", 0))
        self.pipeline_cycles = int(getattr(stats, "pipeline_cycles", 0))
        self.network_cycles = (
            self.vc_wait_cycles
            + self.buffer_wait_cycles
            + self.link_wait_cycles
            + self.pipeline_cycles
        )

        flits = float(getattr(stats, "flits_sent", 0))
        packets = float(getattr(stats, "packets_sent", 0))
        if sim_time_cycles is None:
            sim_time_cycles = max(1, int(self.total_latency_cycles))
        self.network_throughput = flits / max(1.0, float(sim_time_cycles))
        if packets > 0:
            self.network_saturation = min(
                1.0, (self.buffer_wait_cycles + self.vc_wait_cycles) / max(1.0, packets)
            )

    def dse_score(self, weights: dict[str, float] | None = None) -> float:
        if not weights:
            return -float(self.total_latency_cycles)

        score = 0.0
        for metric_name, metric_weight in weights.items():
            if hasattr(self, metric_name):
                score += metric_weight * float(getattr(self, metric_name))
        return score

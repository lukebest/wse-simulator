"""Memory backend interfaces and analytical model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random


class MemoryBackend(ABC):
    @abstractmethod
    def request_cycles(self, size_bytes: int, is_write: bool = False) -> int:
        raise NotImplementedError


@dataclass(slots=True)
class AnalyticalMemoryBackend(MemoryBackend):
    base_latency_cycles: int
    bytes_per_cycle: int
    jitter_model: str = "none"
    jitter_value: float = 0.0
    seed: int = 1234

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def request_cycles(self, size_bytes: int, is_write: bool = False) -> int:
        transfer_cycles = max(1, int(size_bytes / max(self.bytes_per_cycle, 1)))
        jitter = self._sample_jitter()
        return max(1, self.base_latency_cycles + transfer_cycles + jitter)

    def _sample_jitter(self) -> int:
        if self.jitter_model == "none":
            return 0
        if self.jitter_model == "uniform":
            return int(self._rng.uniform(0, max(self.jitter_value, 0.0)))
        if self.jitter_model == "gaussian":
            return int(abs(self._rng.gauss(0, max(self.jitter_value, 0.0))))
        raise ValueError(f"Unsupported jitter_model: {self.jitter_model}")

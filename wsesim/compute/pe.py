"""Processing element performance models."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable


CustomModelFn = Callable[[int, int, int], int]


@dataclass(slots=True)
class PEModel:
    pe_type: str = "systolic"
    width: int = 16
    cube_m_tile: int = 4
    cube_k_tile: int = 32
    cube_n_tile: int = 16
    cube_startup_cycles: int = 27
    cube_steady_cycles: int = 5
    custom_fn: CustomModelFn | None = None

    def matmul_cycles(self, m: int, n: int, k: int) -> int:
        if self.pe_type == "systolic":
            h = self.width
            w = self.width
            return ceil(m / h) * ceil(n / w) * k + h + w - 2
        if self.pe_type == "vector":
            v = max(self.width, 1)
            return m * n * ceil(k / v)
        if self.pe_type == "cube":
            m_tiles = ceil(m / max(self.cube_m_tile, 1))
            n_tiles = ceil(n / max(self.cube_n_tile, 1))
            k_tiles = ceil(k / max(self.cube_k_tile, 1))
            total_tiles = m_tiles * n_tiles * k_tiles
            if total_tiles <= 0:
                return 0
            return self.cube_startup_cycles + (total_tiles - 1) * self.cube_steady_cycles
        if self.pe_type == "custom":
            if self.custom_fn is None:
                raise ValueError("custom_fn must be set when pe_type='custom'.")
            return int(self.custom_fn(m, n, k))
        raise ValueError(f"Unsupported pe_type: {self.pe_type}")

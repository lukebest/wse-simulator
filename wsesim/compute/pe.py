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
    custom_fn: CustomModelFn | None = None

    def matmul_cycles(self, m: int, n: int, k: int) -> int:
        if self.pe_type == "systolic":
            h = self.width
            w = self.width
            return ceil(m / h) * ceil(n / w) * k + h + w - 2
        if self.pe_type == "vector":
            v = max(self.width, 1)
            return m * n * ceil(k / v)
        if self.pe_type == "custom":
            if self.custom_fn is None:
                raise ValueError("custom_fn must be set when pe_type='custom'.")
            return int(self.custom_fn(m, n, k))
        raise ValueError(f"Unsupported pe_type: {self.pe_type}")

"""Workload operation data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GEMMOp:
    name: str
    m: int
    n: int
    k: int
    depends_on: list[str] = field(default_factory=list)
    output_to: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LLMWorkload:
    model_name: str
    ops: list[GEMMOp]
    metadata: dict[str, int | float | str] = field(default_factory=dict)

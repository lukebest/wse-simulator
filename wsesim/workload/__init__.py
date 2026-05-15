"""Workload definitions and mapping."""

from wsesim.workload.generator import generate_moe_decode_ffn_workload
from wsesim.workload.mapper import Mapping, MappingStrategy, NearestNeighborMapping
from wsesim.workload.ops import GEMMOp, LLMWorkload

__all__ = [
    "GEMMOp",
    "LLMWorkload",
    "Mapping",
    "MappingStrategy",
    "NearestNeighborMapping",
    "generate_moe_decode_ffn_workload",
]

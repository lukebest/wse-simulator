"""Workload definitions and mapping."""

from wsesim.workload.generator import generate_moe_decode_ffn_workload
from wsesim.workload.mapper import (
    ExpertLocalityMapping,
    Mapping,
    MappingStrategy,
    NearestNeighborMapping,
)
from wsesim.workload.ops import GEMMOp, LLMWorkload, TokenRoute

__all__ = [
    "GEMMOp",
    "LLMWorkload",
    "TokenRoute",
    "Mapping",
    "MappingStrategy",
    "NearestNeighborMapping",
    "ExpertLocalityMapping",
    "generate_moe_decode_ffn_workload",
]

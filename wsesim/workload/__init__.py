"""Workload definitions and mapping."""

from wsesim.workload.generator import generate_moe_decode_ffn_workload
from wsesim.workload.generator import (
    DeepSeekV3FFNProfile,
    generate_deepseek_v3_decode_ffn_workload,
    generate_moe_decode_ffn_workload,
)
from wsesim.workload.mapper import (
    ExpertAffinityMapping,
    Mapping,
    MappingStrategy,
    NearestNeighborMapping,
)
from wsesim.workload.ops import GEMMOp, LLMWorkload

__all__ = [
    "GEMMOp",
    "LLMWorkload",
    "DeepSeekV3FFNProfile",
    "Mapping",
    "MappingStrategy",
    "NearestNeighborMapping",
    "ExpertAffinityMapping",
    "generate_moe_decode_ffn_workload",
    "generate_deepseek_v3_decode_ffn_workload",
]

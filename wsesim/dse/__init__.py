"""Design-space exploration module."""

from wsesim.dse.engine import DSEEngine, DSETrial
from wsesim.dse.pipeline_analysis import (
    PipelineBreakdown,
    PipelineStage,
    compute_pipeline_breakdown,
)

__all__ = [
    "DSEEngine",
    "DSETrial",
    "PipelineStage",
    "PipelineBreakdown",
    "compute_pipeline_breakdown",
]

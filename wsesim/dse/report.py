"""DSE report helpers."""

from __future__ import annotations

from dataclasses import asdict

from wsesim.core.config import WSEConfig


def summarize_best(history: list[tuple[WSEConfig, float]]) -> dict[str, object]:
    if not history:
        return {"best_score": None, "best_config": None, "trials": 0}
    best_cfg, best_score = max(history, key=lambda item: item[1])
    return {"best_score": best_score, "best_config": asdict(best_cfg), "trials": len(history)}

"""Static defect generation for cores and links."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(slots=True)
class DefectMap:
    dead_cores: set[int]
    dead_links: set[tuple[int, int]]

    @staticmethod
    def generate(
        num_cores: int,
        links: set[tuple[int, int]],
        core_defect_rate: float,
        link_defect_rate: float,
        seed: int,
    ) -> "DefectMap":
        rng = random.Random(seed)
        dead_cores = {
            core for core in range(num_cores) if rng.random() < max(0.0, min(core_defect_rate, 1.0))
        }
        dead_links = {
            link for link in links if rng.random() < max(0.0, min(link_defect_rate, 1.0))
        }
        return DefectMap(dead_cores=dead_cores, dead_links=dead_links)

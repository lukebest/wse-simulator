from __future__ import annotations

from wsesim.fault.defect_map import DefectMap


def test_defect_map_generation_is_bounded() -> None:
    links = {(0, 1), (1, 0), (1, 2)}
    defects = DefectMap.generate(
        num_cores=8,
        links=links,
        core_defect_rate=0.25,
        link_defect_rate=0.5,
        seed=1234,
    )
    assert defects.dead_cores.issubset(set(range(8)))
    assert defects.dead_links.issubset(links)

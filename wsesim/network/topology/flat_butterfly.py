"""Flattened butterfly topology with 6x8-aware grouping."""

from __future__ import annotations

from wsesim.network.topology.base import Topology


class FlatButterfly(Topology):
    def build(self, num_nodes: int) -> dict[int, list[int]]:
        graph_sets: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
        groups = self._build_groups(num_nodes)

        for group in groups:
            for src in group:
                for dst in group:
                    if src != dst:
                        graph_sets[src].add(dst)

        for src_group_idx, src_group in enumerate(groups):
            for dst_group_idx, dst_group in enumerate(groups):
                if src_group_idx == dst_group_idx:
                    continue
                width = min(len(src_group), len(dst_group))
                for lane in range(width):
                    src = src_group[lane]
                    dst = dst_group[lane]
                    if src != dst:
                        graph_sets[src].add(dst)

        return {node: sorted(neighbors) for node, neighbors in graph_sets.items()}

    def _build_groups(self, num_nodes: int) -> list[list[int]]:
        if num_nodes == 48:
            # 6x8 matrix split into six 2x4 blocks.
            return [
                [0, 1, 2, 3, 8, 9, 10, 11],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [16, 17, 18, 19, 24, 25, 26, 27],
                [20, 21, 22, 23, 28, 29, 30, 31],
                [32, 33, 34, 35, 40, 41, 42, 43],
                [36, 37, 38, 39, 44, 45, 46, 47],
            ]

        group_size = max(2, int(num_nodes**0.5))
        groups: list[list[int]] = []
        for start in range(0, num_nodes, group_size):
            groups.append(list(range(start, min(num_nodes, start + group_size))))
        return groups

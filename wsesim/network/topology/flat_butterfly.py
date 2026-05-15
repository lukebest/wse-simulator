"""Simplified flat butterfly topology."""

from __future__ import annotations

from wsesim.network.topology.base import Topology


class FlatButterfly(Topology):
    def build(self, num_nodes: int) -> dict[int, list[int]]:
        # Simplified high-radix approximation: each node links to all nodes in
        # its local group and one representative in other groups.
        group_size = max(2, int(num_nodes**0.5))
        graph: dict[int, list[int]] = {i: [] for i in range(num_nodes)}

        for node in range(num_nodes):
            group = node // group_size
            group_start = group * group_size
            group_end = min(num_nodes, group_start + group_size)
            for peer in range(group_start, group_end):
                if peer != node:
                    graph[node].append(peer)

            representative = node % group_size
            for other_group_start in range(0, num_nodes, group_size):
                if other_group_start == group_start:
                    continue
                target = min(other_group_start + representative, num_nodes - 1)
                if target != node and target not in graph[node]:
                    graph[node].append(target)
        return graph

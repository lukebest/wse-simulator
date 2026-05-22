"""Routing for TDM flattened butterfly overlays on physical mesh links."""

from __future__ import annotations

from dataclasses import dataclass, field

from wsesim.network.routing.base import RoutingAlgorithm
from wsesim.network.topology.tdm_flat_butterfly import TDMFlatButterfly


@dataclass(slots=True)
class _SegmentState:
    endpoint: int
    color: int
    logical_link: tuple[int, int]
    remaining_hops: list[int]


@dataclass(slots=True)
class TDMFlatButterflyRouting(RoutingAlgorithm):
    topology: TDMFlatButterfly
    _packet_state: dict[int, _SegmentState] = field(default_factory=dict)

    def next_hop(self, current: int, dst: int, graph: dict[int, list[int]]) -> int:
        # Fallback for legacy callers that do not expose packet context.
        return self._compute_next_hop(current=current, dst=dst, packet_key=-1)

    def next_hop_with_color(
        self, packet, current: int, dst: int, graph: dict[int, list[int]]
    ) -> tuple[int, int | None, tuple[int, int] | None]:
        del graph
        packet_key = id(packet)
        next_hop = self._compute_next_hop(current=current, dst=dst, packet_key=packet_key)
        state = self._packet_state.get(packet_key)
        if state is None:
            return next_hop, None, None
        if len(state.remaining_hops) == 0:
            self._packet_state.pop(packet_key, None)
        return next_hop, state.color, state.logical_link

    def clear_packet_state(self, packet) -> None:
        self._packet_state.pop(id(packet), None)

    def current_logical_endpoint(self, packet, current: int, dst: int) -> int:
        packet_key = id(packet)
        state = self._packet_state.get(packet_key)
        if state is None:
            endpoint = self._next_endpoint(current=current, dst=dst)
            return endpoint
        return state.endpoint

    def color_for(self, packet, current: int, dst: int) -> int | None:
        packet_key = id(packet)
        state = self._packet_state.get(packet_key)
        if state is None:
            _ = self._compute_next_hop(current=current, dst=dst, packet_key=packet_key)
            state = self._packet_state.get(packet_key)
        return None if state is None else state.color

    def _compute_next_hop(self, current: int, dst: int, packet_key: int) -> int:
        if current == dst:
            return dst

        state = self._packet_state.get(packet_key)
        if state is None:
            endpoint = self._next_endpoint(current=current, dst=dst)
            logical_link = (current, endpoint)
            color = self.topology.coloring().color_of_logical.get(logical_link)
            path = self.topology.physical_path(current, endpoint)
            if not path:
                return endpoint
            state = _SegmentState(
                endpoint=endpoint,
                color=-1 if color is None else color,
                logical_link=logical_link,
                remaining_hops=[hop_dst for _, hop_dst in path],
            )
            self._packet_state[packet_key] = state

        next_hop = state.remaining_hops.pop(0)
        if len(state.remaining_hops) == 0:
            self._packet_state.pop(packet_key, None)
        return next_hop

    def _next_endpoint(self, current: int, dst: int) -> int:
        src_coords = list(self.topology.to_coords(current))
        dst_coords = self.topology.to_coords(dst)
        for dim in range(self.topology.n):
            if src_coords[dim] != dst_coords[dim]:
                src_coords[dim] = dst_coords[dim]
                return self.topology.to_node(tuple(src_coords))
        return dst

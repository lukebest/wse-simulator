"""Packet and flit definitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(slots=True)
class Packet:
    src: int
    dst: int
    size_bytes: int
    payload_type: str
    creation_time: float = 0.0


@dataclass(slots=True)
class Flit:
    packet: Packet
    flit_id: int
    is_head: bool
    is_tail: bool


def packet_to_num_flits(packet: Packet, flit_bytes: int = 32) -> int:
    return max(1, ceil(packet.size_bytes / max(flit_bytes, 1)))

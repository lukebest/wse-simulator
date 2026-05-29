"""Network models for NoC/NoW."""

from wsesim.network.color import Color, ColorPlan
from wsesim.network.color_network import ColorNetwork
from wsesim.network.network import UnifiedNetwork
from wsesim.network.packet import Flit, Packet

__all__ = ["UnifiedNetwork", "ColorNetwork", "ColorPlan", "Color", "Packet", "Flit"]

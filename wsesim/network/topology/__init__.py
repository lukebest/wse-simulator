"""Topology registry."""

from wsesim.network.topology.base import Topology
from wsesim.network.topology.flat_butterfly import FlatButterfly
from wsesim.network.topology.mesh2d import Mesh2D
from wsesim.network.topology.torus2d import Torus2D

__all__ = ["Topology", "Mesh2D", "FlatButterfly", "Torus2D"]

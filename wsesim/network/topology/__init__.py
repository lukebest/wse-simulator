"""Topology registry."""

from wsesim.network.topology.base import Topology
from wsesim.network.topology.flat_butterfly import FlatButterfly
from wsesim.network.topology.mesh2d import Mesh2D
from wsesim.network.topology.butterfly import Butterfly
from wsesim.network.topology.supermesh_alter import SuperMeshAlter
from wsesim.network.topology.supermesh_bi import SuperMeshBi

__all__ = ["Topology", "Mesh2D", "FlatButterfly", "Butterfly", "SuperMeshBi", "SuperMeshAlter"]

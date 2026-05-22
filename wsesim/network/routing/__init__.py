"""Routing algorithms."""

from wsesim.network.routing.base import RoutingAlgorithm
from wsesim.network.routing.dimension_order import DimensionOrderRouting
from wsesim.network.routing.table_based import TableBasedRouting
from wsesim.network.routing.tdm_flat_butterfly import TDMFlatButterflyRouting
from wsesim.network.routing.ugal import UGALRouting

__all__ = [
    "RoutingAlgorithm",
    "DimensionOrderRouting",
    "TableBasedRouting",
    "TDMFlatButterflyRouting",
    "UGALRouting",
]

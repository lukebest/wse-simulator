"""Flow-control strategies."""

from wsesim.network.flow_control.base import FlowControl
from wsesim.network.flow_control.credit_vc import CreditBasedVCFlowControl
from wsesim.network.flow_control.wormhole import WormholeFlowControl

from wsesim.network.flow_control.per_color_credit import PerColorCreditFlowControl

__all__ = [
    "FlowControl",
    "CreditBasedVCFlowControl",
    "WormholeFlowControl",
    "PerColorCreditFlowControl",
]

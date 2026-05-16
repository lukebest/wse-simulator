"""Collective communication expansion into point-to-point transfers."""

from __future__ import annotations

from dataclasses import dataclass

from wsesim.workload.ops import TokenRoute


@dataclass(slots=True)
class P2PTransfer:
    src: int
    dst: int
    size_bytes: int
    collective: str
    token_id: int | None = None
    expert_id: int | None = None


class CollectiveOps:
    @staticmethod
    def broadcast(src: int, dsts: list[int], size_bytes: int) -> list[P2PTransfer]:
        return [P2PTransfer(src=src, dst=dst, size_bytes=size_bytes, collective="broadcast") for dst in dsts]

    @staticmethod
    def allreduce(nodes: list[int], size_bytes: int) -> list[P2PTransfer]:
        transfers: list[P2PTransfer] = []
        for i, src in enumerate(nodes):
            dst = nodes[(i + 1) % len(nodes)]
            transfers.append(P2PTransfer(src=src, dst=dst, size_bytes=size_bytes, collective="allreduce"))
        return transfers

    @staticmethod
    def all_to_all(nodes: list[int], per_node_size: int) -> list[P2PTransfer]:
        transfers: list[P2PTransfer] = []
        for src in nodes:
            for dst in nodes:
                if src != dst:
                    transfers.append(
                        P2PTransfer(src=src, dst=dst, size_bytes=per_node_size, collective="all_to_all")
                    )
        return transfers

    @staticmethod
    def moe_dispatch(
        token_routes: list[TokenRoute],
        token_home_cores: dict[int, int],
        expert_cores: dict[int, int],
        token_bytes: int,
    ) -> list[P2PTransfer]:
        transfers: list[P2PTransfer] = []
        for route in token_routes:
            src = token_home_cores[route.token_id]
            for expert_id in route.selected_experts:
                dst = expert_cores[expert_id]
                transfers.append(
                    P2PTransfer(
                        src=src,
                        dst=dst,
                        size_bytes=token_bytes,
                        collective="moe_dispatch",
                        token_id=route.token_id,
                        expert_id=expert_id,
                    )
                )
        return transfers

    @staticmethod
    def moe_combine(
        token_routes: list[TokenRoute],
        token_home_cores: dict[int, int],
        expert_cores: dict[int, int],
        token_bytes: int,
    ) -> list[P2PTransfer]:
        transfers: list[P2PTransfer] = []
        for route in token_routes:
            dst = token_home_cores[route.token_id]
            for expert_id in route.selected_experts:
                src = expert_cores[expert_id]
                transfers.append(
                    P2PTransfer(
                        src=src,
                        dst=dst,
                        size_bytes=token_bytes,
                        collective="moe_combine",
                        token_id=route.token_id,
                        expert_id=expert_id,
                    )
                )
        return transfers

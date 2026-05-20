"""Random search strategy."""

from __future__ import annotations

from copy import deepcopy
import random

from wsesim.core.config import WSEConfig
from wsesim.dse.search.base import SearchStrategy


class RandomSearch(SearchStrategy):
    def __init__(
        self,
        base: WSEConfig,
        seed: int = 1234,
        partition_strategies: list[str] | None = None,
    ) -> None:
        self.base = base
        self._rng = random.Random(seed)
        self._partition_strategies = partition_strategies or [
            "expert",
            "col",
            "k_split",
        ]

    def suggest(self, history: list[tuple[WSEConfig, float]]) -> WSEConfig:
        cfg = deepcopy(self.base)
        cfg.workload.decode_tokens = self._rng.choice([4, 16])
        cfg.workload.partition_strategy = self._rng.choice(self._partition_strategies)
        cfg.workload.partition_shards = self._rng.choice([1, 2, 4, 7])
        cfg.workload.tile_pipeline = self._rng.choice([True, False])

        if cfg.workload.partition_strategy == "expert":
            cfg.workload.partition_shards = 1
        elif cfg.workload.partition_strategy == "row":
            cfg.workload.partition_shards = 1
        elif cfg.workload.partition_strategy == "block":
            cfg.workload.partition_shards = self._rng.choice([1, 4])

        cfg.network.noc.topology = self._rng.choice(
            ["mesh2d", "flat_butterfly", "butterfly", "supermesh_bi", "supermesh_alter"]
        )
        cfg.network.noc.routing = self._rng.choice(["xy", "ugal", "table_based"])
        cfg.network.noc.flow_control = self._rng.choice(["credit_vc", "wormhole"])
        cfg.network.noc.buffer_depth = self._rng.choice([4, 8, 16])
        cfg.network.noc.num_vcs = self._rng.choice([1, 2, 4])
        cfg.network.noc.link_bw_flits_per_cycle = self._rng.choice([1, 2, 4])

        cfg.network.now.topology = self._rng.choice(["mesh2d", "flat_butterfly", "butterfly"])
        cfg.network.now.routing = self._rng.choice(["xy", "ugal", "table_based"])
        cfg.network.now.flow_control = self._rng.choice(["credit_vc", "wormhole"])

        max_gateways = max(1, min(cfg.wafer.cores_per_reticle, 8))
        gateway_candidates = sorted({1, 2, 4, max_gateways})
        cfg.network.gateways_per_reticle = self._rng.choice(gateway_candidates)
        cfg.network.gateway_policy = self._rng.choice(["nearest", "load_aware"])
        cfg.network.io_distribution_policy = self._rng.choice(
            ["round_robin", "nearest", "load_aware"]
        )

        active_experts = min(
            cfg.workload.top_k + cfg.workload.num_shared_experts,
            cfg.workload.num_routed_experts + cfg.workload.num_shared_experts,
        )
        max_shards = max(1, cfg.wafer.total_cores // max(1, active_experts))
        cfg.workload.partition_shards = min(cfg.workload.partition_shards, max_shards)
        return cfg

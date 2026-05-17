from __future__ import annotations

import csv
import json

from wsesim.core.config import WSEConfig
from wsesim.core.stats import SimResult
from wsesim.dse.engine import DSEEngine
from wsesim.dse.report import (
    export_pareto_csv,
    export_trials_csv,
    export_trials_json,
    pareto_front,
    summarize_best,
    summarize_best_trial,
)
from wsesim.dse.search.random import RandomSearch


def _evaluate(cfg: WSEConfig) -> SimResult:
    congestion_scale = max(1, 8 // max(cfg.network.noc.num_vcs, 1))
    return SimResult(
        total_latency_cycles=max(1, 1000 // max(cfg.network.noc.link_bw_flits_per_cycle, 1)),
        vc_wait_cycles=5 * congestion_scale,
        buffer_wait_cycles=3 * congestion_scale,
        link_wait_cycles=2 * congestion_scale,
        network_throughput=float(cfg.network.noc.link_bw_flits_per_cycle),
    )


def test_dse_runs_and_reports_best() -> None:
    base = WSEConfig()
    dse = DSEEngine(base_config=base, strategy=RandomSearch(base, seed=1), evaluator=_evaluate)
    history = dse.run(trials=4)
    summary = summarize_best(history)
    assert len(history) == 4
    assert summary["best_score"] is not None


def test_dse_run_detailed_and_pareto_front() -> None:
    base = WSEConfig()
    dse = DSEEngine(base_config=base, strategy=RandomSearch(base, seed=2), evaluator=_evaluate)
    trials = dse.run_detailed(trials=6)
    summary = summarize_best_trial(trials)
    front = pareto_front(trials)

    assert len(trials) == 6
    assert summary["best_score"] is not None
    assert 1 <= len(front) <= len(trials)
    assert all(trial in trials for trial in front)


def test_dse_score_uses_congestion_penalties() -> None:
    weights = {
        "total_latency_cycles": -1.0,
        "vc_wait_cycles": -2.0,
        "buffer_wait_cycles": -1.0,
        "link_wait_cycles": -1.0,
    }
    low_congestion = SimResult(
        total_latency_cycles=100,
        vc_wait_cycles=1,
        buffer_wait_cycles=1,
        link_wait_cycles=1,
    )
    high_congestion = SimResult(
        total_latency_cycles=100,
        vc_wait_cycles=20,
        buffer_wait_cycles=20,
        link_wait_cycles=20,
    )
    assert low_congestion.dse_score(weights) > high_congestion.dse_score(weights)


def test_dse_exports_json_and_csv(tmp_path) -> None:
    base = WSEConfig()
    dse = DSEEngine(base_config=base, strategy=RandomSearch(base, seed=3), evaluator=_evaluate)
    trials = dse.run_detailed(trials=5)
    front = pareto_front(trials)

    json_path = export_trials_json(trials, tmp_path / "trials.json")
    csv_path = export_trials_csv(trials, tmp_path / "trials.csv")
    pareto_path = export_pareto_csv(front, tmp_path / "pareto.csv")

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(json_payload) == len(trials)
    assert "score" in json_payload[0]
    assert "config" in json_payload[0]
    assert "result" in json_payload[0]

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(trials)
    assert "total_latency_cycles" in rows[0]
    assert "cube_steady_cycles" in rows[0]
    assert "batch_size" in rows[0]
    assert "partition_strategy" in rows[0]
    assert "noc_topology" in rows[0]
    assert "now_routing" in rows[0]
    assert "allreduce_cycles" in rows[0]
    assert "gateway_noc_hops" in rows[0]
    assert "gateway_policy" in rows[0]

    with pareto_path.open("r", encoding="utf-8", newline="") as f:
        pareto_rows = list(csv.DictReader(f))
    assert len(pareto_rows) == len(front)


def test_random_search_explores_dse_dimensions() -> None:
    base = WSEConfig()
    search = RandomSearch(base, seed=42)
    history: list[tuple[WSEConfig, float]] = []

    policies = set()
    io_policies = set()
    gateway_counts = set()
    batch_sizes = set()
    partition_strategies = set()
    noc_topologies = set()
    now_topologies = set()
    tile_pipelines = set()
    for _ in range(80):
        cfg = search.suggest(history)
        policies.add(cfg.network.gateway_policy)
        io_policies.add(cfg.network.io_distribution_policy)
        gateway_counts.add(cfg.network.gateways_per_reticle)
        batch_sizes.add(cfg.workload.decode_tokens)
        partition_strategies.add(cfg.workload.partition_strategy)
        noc_topologies.add(cfg.network.noc.topology)
        now_topologies.add(cfg.network.now.topology)
        tile_pipelines.add(cfg.workload.tile_pipeline)
        if cfg.workload.partition_strategy == "expert":
            assert cfg.workload.partition_shards == 1
        if cfg.workload.partition_strategy == "row":
            assert cfg.workload.partition_shards == 1
        if cfg.workload.partition_strategy == "block":
            assert cfg.workload.partition_shards in {1, 4}
        history.append((cfg, 0.0))

    assert "nearest" in policies
    assert "load_aware" in policies
    assert "round_robin" in io_policies
    assert "nearest" in io_policies
    assert "load_aware" in io_policies
    assert batch_sizes == {4, 16}
    assert "expert" in partition_strategies
    assert "col" in partition_strategies
    assert "k_split" in partition_strategies
    assert "row" in partition_strategies
    assert "block" in partition_strategies
    assert "hybrid_nk" in partition_strategies
    assert "entwined_ring" in partition_strategies
    assert "streaming" in partition_strategies
    assert "mesh2d" in noc_topologies
    assert "flat_butterfly" in noc_topologies
    assert "mesh2d" in now_topologies
    assert "flat_butterfly" in now_topologies
    assert len(gateway_counts) >= 2
    assert True in tile_pipelines
    assert False in tile_pipelines

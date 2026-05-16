"""Configuration dataclasses for WSE simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WaferConfig:
    reticles_x: int = 2
    reticles_y: int = 2
    reticle_rows: int = 6
    reticle_cols: int = 8
    reticle_dead_positions: tuple[tuple[int, int], ...] = ((3, 0), (4, 0))
    reticle_io_positions: tuple[tuple[int, int], ...] = ((1, 0), (2, 0))
    io_bandwidth_gbps: float = 256.0
    defect_rate: float = 0.0
    defect_seed: int = 42

    @property
    def cores_per_reticle(self) -> int:
        grid_nodes = max(1, self.reticle_rows) * max(1, self.reticle_cols)
        dead_nodes = len(self.reticle_dead_positions)
        io_nodes = len(self.reticle_io_positions)
        return max(1, grid_nodes - dead_nodes - io_nodes)

    @property
    def reticle_count(self) -> int:
        return self.reticles_x * self.reticles_y

    @property
    def total_cores(self) -> int:
        return self.reticle_count * self.cores_per_reticle


@dataclass(slots=True)
class ComputeConfig:
    pe_type: str = "cube"
    pe_width: int = 16
    pe_freq_ghz: float = 2.0
    l1_capacity_kb: int = 2048
    l1_read_bw_gbps: float = 1024.0
    l1_write_bw_gbps: float = 1024.0
    l1_latency_cycles: int = 2
    cube_m_tile: int = 4
    cube_k_tile: int = 32
    cube_n_tile: int = 16
    cube_startup_cycles: int = 27
    cube_steady_cycles: int = 5


@dataclass(slots=True)
class NetworkDomainConfig:
    topology: str = "mesh2d"
    routing: str = "xy"
    flow_control: str = "credit_vc"
    freq_ghz: float = 2.2
    link_width_bytes: int = 128
    link_bw_flits_per_cycle: int = 1
    link_latency_cycles: int = 1
    num_vcs: int = 2
    buffer_depth: int = 8
    router_pipeline_mode: str = "4_stage"
    rc_latency_cycles: int = 1
    va_latency_cycles: int = 1
    sa_latency_cycles: int = 1
    st_latency_cycles: int = 1
    crossbar_bw_flits_per_cycle: int = 1


@dataclass(slots=True)
class NetworkSetConfig:
    noc: NetworkDomainConfig = field(default_factory=NetworkDomainConfig)
    now: NetworkDomainConfig = field(
        default_factory=lambda: NetworkDomainConfig(link_latency_cycles=5)
    )
    gateways_per_reticle: int = 1
    gateway_policy: str = "nearest"
    io_distribution_policy: str = "round_robin"


@dataclass(slots=True)
class MemoryConfig:
    num_controllers: int = 4
    total_capacity_gb: float = 64.0
    peak_bandwidth_gbps: float = 900.0
    base_latency_ns: float = 120.0
    per_core_bandwidth_gbps: float = 256.0
    per_core_latency_ns: float = 100.0
    jitter_model: str = "none"
    jitter_value: float = 0.0


@dataclass(slots=True)
class WorkloadConfig:
    model_name: str = "deepseek_v4_pro_ffn_decode"
    hidden_dim: int = 7168
    expert_ffn_dim: int = 3072
    num_experts: int = 8
    num_routed_experts: int = 384
    num_shared_experts: int = 1
    top_k: int = 6
    decode_batch_size: int = 1
    decode_tokens: int = 32
    routing_skew_alpha: float = 1.2
    capacity_factor: float = 1.25
    partition_strategy: str = "expert"
    partition_shards: int = 1
    collective_algorithm: str = "ring"
    mapping_strategy: str = "nearest_neighbor"


@dataclass(slots=True)
class DSEConfig:
    objective: str = "latency_min"
    max_trials: int = 50
    random_seed: int = 1234
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "total_latency_cycles": -1.0,
            "vc_wait_cycles": -0.1,
            "buffer_wait_cycles": -0.1,
            "link_wait_cycles": -0.05,
            "gateway_noc_hops": -0.001,
            "gateway_peak_load": -0.05,
            "memory_stall_cycles": -0.5,
            "io_injection_cycles": -0.2,
        }
    )


@dataclass(slots=True)
class WSEConfig:
    wafer: WaferConfig = field(default_factory=WaferConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    network: NetworkSetConfig = field(default_factory=NetworkSetConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    dse: DSEConfig = field(default_factory=DSEConfig)

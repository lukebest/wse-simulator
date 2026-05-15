"""Configuration dataclasses for WSE simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WaferConfig:
    reticles_x: int = 2
    reticles_y: int = 2
    cores_per_reticle: int = 16
    defect_rate: float = 0.0
    defect_seed: int = 42

    @property
    def reticle_count(self) -> int:
        return self.reticles_x * self.reticles_y

    @property
    def total_cores(self) -> int:
        return self.reticle_count * self.cores_per_reticle


@dataclass(slots=True)
class ComputeConfig:
    pe_type: str = "systolic"
    pe_width: int = 16
    pe_freq_ghz: float = 1.0
    l1_capacity_kb: int = 256
    l1_read_bw_gbps: float = 1024.0
    l1_write_bw_gbps: float = 1024.0
    l1_latency_cycles: int = 2


@dataclass(slots=True)
class NetworkDomainConfig:
    topology: str = "mesh2d"
    routing: str = "xy"
    flow_control: str = "credit_vc"
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


@dataclass(slots=True)
class MemoryConfig:
    num_controllers: int = 4
    total_capacity_gb: float = 64.0
    peak_bandwidth_gbps: float = 900.0
    base_latency_ns: float = 120.0
    jitter_model: str = "none"
    jitter_value: float = 0.0


@dataclass(slots=True)
class WorkloadConfig:
    model_name: str = "mixtral_8x7b"
    hidden_dim: int = 4096
    expert_ffn_dim: int = 14336
    num_experts: int = 8
    top_k: int = 2
    decode_batch_size: int = 1
    decode_tokens: int = 32
    partition_strategy: str = "expert"
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

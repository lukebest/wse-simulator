"""Ring allreduce traffic generation for SimPy network simulation."""

from __future__ import annotations

from math import ceil


def generate_ring_allreduce_traffic(
    participating_nodes: list[int],
    payload_bytes_per_expert: int,
    num_experts: int,
    strategy: str = "sequential",
) -> list[dict]:
    """Generate packet-level traffic for a ring allreduce.

    Two-phase ring: reduce-scatter (S-1 steps) then all-gather (S-1 steps).
    Each step sends one chunk from node[i] to node[(i+1) % S].

    Parameters
    ----------
    participating_nodes : physical node IDs forming the ring.
    payload_bytes_per_expert : total bytes each expert must allreduce.
    num_experts : how many experts run allreduce concurrently.
    strategy : "sequential" — all experts inject simultaneously;
               "entwined" — expert ring steps are interleaved with
               staggered delays so different experts use different
               ring links at the same time.

    Returns a list of traffic dicts compatible with
    ``_run_hierarchical_network_simulation``.
    """
    S = len(participating_nodes)
    if S <= 1 or num_experts <= 0 or payload_bytes_per_expert <= 0:
        return []

    chunk_bytes = max(1, ceil(payload_bytes_per_expert / S))
    total_steps = 2 * (S - 1)
    traffic: list[dict] = []

    if strategy == "entwined":
        traffic = _entwined_ring(participating_nodes, chunk_bytes, total_steps, num_experts)
    else:
        traffic = _sequential_ring(participating_nodes, chunk_bytes, total_steps, num_experts)

    return traffic


def _sequential_ring(
    nodes: list[int], chunk_bytes: int, total_steps: int, num_experts: int,
) -> list[dict]:
    """All experts inject their ring steps concurrently (no stagger)."""
    S = len(nodes)
    traffic: list[dict] = []
    for expert_id in range(num_experts):
        for step in range(total_steps):
            phase = "allreduce_rs" if step < S - 1 else "allreduce_ag"
            for i in range(S):
                src = nodes[i]
                dst = nodes[(i + 1) % S]
                traffic.append({
                    "src_core": src,
                    "dst_core": dst,
                    "src_io_phys": None,
                    "dst_io_phys": None,
                    "size_bytes": chunk_bytes,
                    "payload": phase,
                    "delay_cycles": 0,
                })
    return traffic


def _entwined_ring(
    nodes: list[int], chunk_bytes: int, total_steps: int, num_experts: int,
) -> list[dict]:
    """Interleave expert ring steps with staggered offsets.

    Expert ``e`` at ring step ``s`` starts at delay proportional to
    ``e * step_interval``.  Different experts' packets therefore occupy
    different ring links simultaneously, reducing peak contention.
    The actual communication savings emerge from the SimPy simulation
    when these staggered packets share the network.
    """
    S = len(nodes)
    single_hop_estimate = max(1, chunk_bytes // 128 + 4 + 1)
    step_interval = max(1, single_hop_estimate // max(1, num_experts))

    traffic: list[dict] = []
    for step in range(total_steps):
        phase = "allreduce_rs" if step < S - 1 else "allreduce_ag"
        for expert_id in range(num_experts):
            base_delay = step * single_hop_estimate + expert_id * step_interval
            for i in range(S):
                src = nodes[(i + expert_id) % S]
                dst = nodes[(i + expert_id + 1) % S]
                traffic.append({
                    "src_core": src,
                    "dst_core": dst,
                    "src_io_phys": None,
                    "dst_io_phys": None,
                    "size_bytes": chunk_bytes,
                    "payload": phase,
                    "delay_cycles": base_delay,
                })
    return traffic

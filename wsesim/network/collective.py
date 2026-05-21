"""Collective traffic generation for SimPy network simulation."""

from __future__ import annotations

from math import ceil, isqrt, log2


COLLECTIVE_ALGORITHMS = {
    "ring",
    "recursive_halving_doubling",
    "2d_ring",
    "direct_allgather",
    "hierarchical",
}


def select_collective_algorithm(
    partition_strategy: str,
    noc_topology: str,
    now_topology: str,
    shards: int,
    cores_per_reticle: int,
    reticle_count: int,
) -> str:
    """Select a collective algorithm based on topology and scale."""
    del partition_strategy, now_topology
    s = max(1, shards)
    if s > max(1, cores_per_reticle) and reticle_count > 1:
        return "hierarchical"
    if s <= 8:
        return "direct_allgather"
    if noc_topology in {"butterfly", "flat_butterfly"} and _is_power_of_two(s):
        return "recursive_halving_doubling"
    if noc_topology == "mesh2d":
        rows, cols = _factor_near_square(s)
        if rows > 1 and cols > 1 and rows * cols == s:
            return "2d_ring"
    return "ring"


def generate_collective_traffic(
    algorithm: str,
    participating_nodes_global: list[int],
    cores_per_reticle: int,
    payload_bytes_per_expert: int,
    num_experts: int,
    *,
    topology_hint: dict | None = None,
    ring_strategy: str = "sequential",
) -> list[dict]:
    """Generate packet traffic for collective communication."""
    s = len(participating_nodes_global)
    if s <= 1 or payload_bytes_per_expert <= 0 or num_experts <= 0:
        return []
    algo = (algorithm or "ring").strip().lower()
    if algo not in COLLECTIVE_ALGORITHMS:
        algo = "ring"

    if algo == "ring":
        total_steps = 2 * (s - 1)
        chunk_bytes = max(1, ceil(payload_bytes_per_expert / s))
        if ring_strategy == "entwined":
            return _entwined_ring(participating_nodes_global, chunk_bytes, total_steps, num_experts)
        return _sequential_ring(participating_nodes_global, chunk_bytes, total_steps, num_experts)

    if algo == "recursive_halving_doubling":
        return _recursive_halving_doubling(
            participating_nodes_global, payload_bytes_per_expert, num_experts
        )
    if algo == "2d_ring":
        rows = int((topology_hint or {}).get("rows", 0))
        cols = int((topology_hint or {}).get("cols", 0))
        if rows * cols != s:
            rows, cols = _factor_near_square(s)
        return _2d_ring(participating_nodes_global, payload_bytes_per_expert, num_experts, rows, cols)
    if algo == "direct_allgather":
        return _direct_allgather(participating_nodes_global, payload_bytes_per_expert, num_experts)
    return _hierarchical(participating_nodes_global, payload_bytes_per_expert, num_experts, cores_per_reticle)


def generate_ring_allreduce_traffic(
    participating_nodes: list[int],
    payload_bytes_per_expert: int,
    num_experts: int,
    strategy: str = "sequential",
) -> list[dict]:
    """Backward-compatible wrapper for ring allreduce generation."""
    return generate_collective_traffic(
        algorithm="ring",
        participating_nodes_global=participating_nodes,
        cores_per_reticle=max(1, len(participating_nodes)),
        payload_bytes_per_expert=payload_bytes_per_expert,
        num_experts=num_experts,
        ring_strategy=strategy,
    )


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
                traffic.append(_packet(src, dst, chunk_bytes, phase, 0))
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
                traffic.append(_packet(src, dst, chunk_bytes, phase, base_delay))
    return traffic


def _recursive_halving_doubling(nodes: list[int], payload_bytes: int, num_experts: int) -> list[dict]:
    s = len(nodes)
    if not _is_power_of_two(s):
        total_steps = 2 * (s - 1)
        chunk = max(1, ceil(payload_bytes / s))
        return _sequential_ring(nodes, chunk, total_steps, num_experts)

    chunk = max(1, ceil(payload_bytes / s))
    stages = int(log2(s))
    traffic: list[dict] = []
    for expert_id in range(num_experts):
        for stage in range(stages):
            stride = 1 << stage
            delay = expert_id * stages * 2 + stage
            for idx in range(s):
                partner = idx ^ stride
                src = nodes[idx]
                dst = nodes[partner]
                traffic.append(_packet(src, dst, chunk, "allreduce_rs", delay))
        for stage in range(stages):
            stride = 1 << (stages - stage - 1)
            delay = expert_id * stages * 2 + stages + stage
            for idx in range(s):
                partner = idx ^ stride
                src = nodes[idx]
                dst = nodes[partner]
                traffic.append(_packet(src, dst, chunk, "allreduce_ag", delay))
    return traffic


def _2d_ring(
    nodes: list[int], payload_bytes: int, num_experts: int, rows: int, cols: int
) -> list[dict]:
    if rows <= 0 or cols <= 0 or rows * cols != len(nodes):
        rows, cols = _factor_near_square(len(nodes))
    matrix = [nodes[r * cols : (r + 1) * cols] for r in range(rows)]
    total_steps_row = 2 * (cols - 1)
    total_steps_col = 2 * (rows - 1)
    chunk = max(1, ceil(payload_bytes / len(nodes)))
    traffic: list[dict] = []
    for expert_id in range(num_experts):
        start_idx = len(traffic)
        for row_idx in range(rows):
            row_nodes = matrix[row_idx]
            traffic.extend(_sequential_ring(row_nodes, chunk, total_steps_row, 1))
        for col_idx in range(cols):
            col_nodes = [matrix[row][col_idx] for row in range(rows)]
            traffic.extend(_sequential_ring(col_nodes, chunk, total_steps_col, 1))
        # Add offset to keep experts staggered in time.
        for pkt in traffic[start_idx:]:
            pkt["delay_cycles"] += expert_id * max(1, rows + cols)
    return traffic


def _direct_allgather(nodes: list[int], payload_bytes: int, num_experts: int) -> list[dict]:
    s = len(nodes)
    chunk = max(1, ceil(payload_bytes / s))
    traffic: list[dict] = []
    for expert_id in range(num_experts):
        base = expert_id * s
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                traffic.append(_packet(src, dst, chunk, "allgather", base))
    return traffic


def _hierarchical(
    nodes: list[int], payload_bytes: int, num_experts: int, cores_per_reticle: int
) -> list[dict]:
    groups: dict[int, list[int]] = {}
    for node in nodes:
        reticle = node // max(1, cores_per_reticle)
        groups.setdefault(reticle, []).append(node)
    reticle_nodes = [groups[k] for k in sorted(groups)]
    leaders = [group[0] for group in reticle_nodes if group]
    if len(leaders) <= 1:
        total_steps = 2 * (len(nodes) - 1)
        chunk = max(1, ceil(payload_bytes / len(nodes)))
        return _sequential_ring(nodes, chunk, total_steps, num_experts)

    traffic: list[dict] = []
    local_chunk = max(1, ceil(payload_bytes / len(nodes)))
    leader_chunk = max(1, ceil(payload_bytes / len(leaders)))
    for expert_id in range(num_experts):
        stage_offset = expert_id * 32
        # Stage 1: intra-reticle reduce-scatter style ring.
        for group in reticle_nodes:
            if len(group) <= 1:
                continue
            for step in range(len(group) - 1):
                for idx, src in enumerate(group):
                    dst = group[(idx + 1) % len(group)]
                    traffic.append(_packet(src, dst, local_chunk, "allreduce_rs", stage_offset + step))
        stage_offset += max(1, max(len(g) for g in reticle_nodes) - 1)
        # Stage 2: inter-reticle allreduce among leaders.
        total_steps = 2 * (len(leaders) - 1)
        for step in range(total_steps):
            phase = "allreduce_rs" if step < len(leaders) - 1 else "allreduce_ag"
            for idx, src in enumerate(leaders):
                dst = leaders[(idx + 1) % len(leaders)]
                traffic.append(_packet(src, dst, leader_chunk, phase, stage_offset + step))
        stage_offset += total_steps
        # Stage 3: intra-reticle broadcast from leaders.
        for group in reticle_nodes:
            leader = group[0]
            for dst in group[1:]:
                traffic.append(_packet(leader, dst, local_chunk, "allgather", stage_offset))
    return traffic


def _packet(src: int, dst: int, size_bytes: int, payload: str, delay_cycles: int) -> dict:
    return {
        "src_core": src,
        "dst_core": dst,
        "src_io_phys": None,
        "dst_io_phys": None,
        "size_bytes": max(1, int(size_bytes)),
        "payload": payload,
        "delay_cycles": max(0, int(delay_cycles)),
    }


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _factor_near_square(value: int) -> tuple[int, int]:
    side = max(1, isqrt(max(1, value)))
    for rows in range(side, 0, -1):
        if value % rows == 0:
            return rows, value // rows
    return 1, max(1, value)

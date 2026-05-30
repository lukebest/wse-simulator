"""Collective traffic generation for SimPy network simulation."""

from __future__ import annotations

from math import ceil, isqrt, log2


COLLECTIVE_ALGORITHMS = {
    "ring",
    "recursive_halving_doubling",
    "2d_ring",
    "direct_allgather",
    "hierarchical",
    "broadcast_tree",
    "reduction_tree",
    "all_to_all",
    "systolic",
    "mixed",
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
    if algo == "broadcast_tree":
        return _broadcast_tree(participating_nodes_global, payload_bytes_per_expert, num_experts, topology_hint)
    if algo == "reduction_tree":
        return _reduction_tree(participating_nodes_global, payload_bytes_per_expert, num_experts, topology_hint)
    if algo == "all_to_all":
        return _all_to_all(participating_nodes_global, payload_bytes_per_expert, num_experts)
    if algo == "systolic":
        return _systolic_stream(participating_nodes_global, payload_bytes_per_expert, num_experts, topology_hint)
    if algo == "mixed":
        return generate_mixed_taxonomy_traffic(
            participating_nodes_global,
            payload_bytes_per_expert,
            num_experts,
            topology_hint=topology_hint,
        )
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


def _broadcast_tree(
    nodes: list[int], payload_bytes: int, num_experts: int, topology_hint: dict | None
) -> list[dict]:
    rows, cols = _mesh_hint(nodes, topology_hint)
    chunk = max(1, ceil(payload_bytes / len(nodes)))
    traffic: list[dict] = []
    for expert_id in range(num_experts):
        base = expert_id * cols
        for row in range(rows):
            root = row * cols
            for c in range(1, cols):
                dst = row * cols + c
                traffic.append(_packet(root, dst, chunk, "broadcast", base + c))
    return traffic


def _reduction_tree(
    nodes: list[int], payload_bytes: int, num_experts: int, topology_hint: dict | None
) -> list[dict]:
    rows, cols = _mesh_hint(nodes, topology_hint)
    chunk = max(1, ceil(payload_bytes / len(nodes)))
    traffic: list[dict] = []
    for expert_id in range(num_experts):
        base = expert_id * rows
        for col in range(cols):
            root = col
            for r in range(1, rows):
                src = r * cols + col
                traffic.append(_packet(src, root, chunk, "reduce", base + r))
    return traffic


def _all_to_all(nodes: list[int], payload_bytes: int, num_experts: int) -> list[dict]:
    s = len(nodes)
    chunk = max(1, ceil(payload_bytes / s))
    traffic: list[dict] = []
    for expert_id in range(num_experts):
        for phase in range(s - 1):
            for idx, src in enumerate(nodes):
                dst = nodes[(idx + phase + 1) % s]
                if src != dst:
                    traffic.append(
                        _packet(src, dst, chunk, "all_to_all", expert_id * s * s + phase * s + idx)
                    )
    return traffic


def _systolic_stream(
    nodes: list[int], payload_bytes: int, num_experts: int, topology_hint: dict | None
) -> list[dict]:
    rows, cols = _mesh_hint(nodes, topology_hint)
    chunk = max(1, ceil(payload_bytes / max(1, cols)))
    traffic: list[dict] = []
    for expert_id in range(num_experts):
        for row in range(rows):
            for c in range(cols - 1):
                src = row * cols + c
                dst = row * cols + (c + 1)
                traffic.append(_packet(src, dst, chunk, "systolic", expert_id * rows + row + c))
    return traffic


def generate_mixed_taxonomy_traffic(
    nodes: list[int],
    payload_bytes: int,
    num_experts: int = 1,
    *,
    topology_hint: dict | None = None,
) -> list[dict]:
    """Combined NN communication patterns with staggered phases."""
    traffic: list[dict] = []
    offset = 0
    for gen, phase in (
        (_systolic_stream, "systolic"),
        (_reduction_tree, "reduce"),
        (_broadcast_tree, "broadcast"),
        (_sequential_ring, "allreduce"),
        (_direct_allgather, "allgather"),
        (_all_to_all, "all_to_all"),
    ):
        if gen is _sequential_ring:
            s = len(nodes)
            chunk = max(1, ceil(payload_bytes / s))
            part = gen(nodes, chunk, 2 * (s - 1), num_experts)
        elif gen in (_broadcast_tree, _reduction_tree, _systolic_stream):
            part = gen(nodes, payload_bytes, num_experts, topology_hint)
        else:
            part = gen(nodes, payload_bytes, num_experts)
        for pkt in part:
            pkt["delay_cycles"] += offset
            pkt["payload"] = f"mixed_{pkt['payload']}"
        traffic.extend(part)
        offset += max(64, len(part))
    return traffic


def default_color_map() -> dict[str, int]:
    """Map payload types to colors in the mixed plan."""
    return {
        "systolic": 0,
        "mixed_systolic": 0,
        "reduce": 1,
        "mixed_reduce": 1,
        "broadcast": 0,
        "mixed_broadcast": 0,
        "allreduce_rs": 0,
        "allreduce_ag": 0,
        "mixed_allreduce_rs": 0,
        "mixed_allreduce_ag": 0,
        "allgather": 0,
        "mixed_allgather": 0,
        "all_to_all": 1,
        "mixed_all_to_all": 1,
    }


def _mesh_hint(nodes: list[int], topology_hint: dict | None) -> tuple[int, int]:
    hint = topology_hint or {}
    rows = int(hint.get("rows", 0))
    cols = int(hint.get("cols", 0))
    if rows * cols == len(nodes):
        return rows, cols
    return _factor_near_square(len(nodes))


# ---------------------------------------------------------------------------
# Ideal-routing collective workloads (broadcast / gather / reduce /
# allreduce / allgather). Each pattern has two realisations:
#   * baseline -- naive *direct* algorithm, single VN (color 0), XY routing
#   * ideal    -- minimal-link spanning tree / 2D ring on the mesh-independent
#                 color catalog (wsesim.network.color_catalog), with
#                 level-based delays for pipelining.
# ---------------------------------------------------------------------------

from wsesim.network.color_catalog import (  # noqa: E402
    C_AG_COL_NORTH,
    C_AG_COL_SOUTH,
    C_AG_ROW_EAST,
    C_AG_ROW_WEST,
    C_AR_AG_COL_NORTH,
    C_AR_AG_ROW_WEST,
    C_AR_RS_COL_SOUTH,
    C_AR_RS_ROW_EAST,
    C_BCAST_COL_SOUTH,
    C_BCAST_ROW_EAST,
    C_GATHER_COL_NORTH,
    C_GATHER_ROW_WEST,
    C_REDUCE_COL_NORTH,
    C_REDUCE_ROW_WEST,
)

IDEAL_PATTERNS = ["broadcast", "gather", "reduce", "allreduce", "allgather"]


def _cedge(src: int, dst: int, size_bytes: int, payload: str, delay: int, color: int) -> dict:
    pkt = _packet(src, dst, size_bytes, payload, delay)
    pkt["color"] = int(color)
    return pkt


def generate_baseline_collective(
    pattern: str, rows: int, cols: int, chunk_bytes: int
) -> list[dict]:
    """Naive direct realisation on a single VN (all color 0, XY routing)."""
    n = rows * cols
    nodes = list(range(n))
    root = 0
    chunk = max(1, int(chunk_bytes))
    pattern = pattern.lower()

    if pattern == "broadcast":
        return [_cedge(root, d, chunk, "broadcast", 0, 0) for d in nodes if d != root]
    if pattern in ("gather", "reduce"):
        return [_cedge(s, root, chunk, pattern, 0, 0) for s in nodes if s != root]
    if pattern == "allreduce":
        # reduce-to-root then broadcast-from-root
        pk = [_cedge(s, root, chunk, "allreduce", 0, 0) for s in nodes if s != root]
        pk += [_cedge(root, d, chunk, "allreduce", 1, 0) for d in nodes if d != root]
        return pk
    if pattern == "allgather":
        # direct all-to-all
        return [
            _cedge(s, d, chunk, "allgather", 0, 0)
            for s in nodes
            for d in nodes
            if s != d
        ]
    raise ValueError(f"Unknown pattern {pattern!r}")


def generate_ideal_collective(
    pattern: str, rows: int, cols: int, chunk_bytes: int
) -> list[dict]:
    """Ideal static-route realisation on the color catalog."""
    chunk = max(1, int(chunk_bytes))
    pattern = pattern.lower()
    if pattern == "broadcast":
        return _ideal_broadcast(rows, cols, chunk)
    if pattern == "gather":
        return _ideal_tree_to_root(rows, cols, chunk, "gather", C_GATHER_COL_NORTH, C_GATHER_ROW_WEST)
    if pattern == "reduce":
        return _ideal_tree_to_root(rows, cols, chunk, "reduce", C_REDUCE_COL_NORTH, C_REDUCE_ROW_WEST)
    if pattern == "allgather":
        return _ideal_allgather(rows, cols, chunk)
    if pattern == "allreduce":
        return _ideal_allreduce(rows, cols, chunk)
    raise ValueError(f"Unknown pattern {pattern!r}")


def _ideal_broadcast(rows: int, cols: int, chunk: int) -> list[dict]:
    """Spanning tree from corner (0,0): row-0 east ripple, then per-column south."""
    pk: list[dict] = []
    for c in range(cols - 1):
        pk.append(_cedge(c, c + 1, chunk, "broadcast", c, C_BCAST_ROW_EAST))
    for c in range(cols):
        for r in range(rows - 1):
            delay = c + 1 + r
            pk.append(
                _cedge(r * cols + c, (r + 1) * cols + c, chunk, "broadcast", delay, C_BCAST_COL_SOUTH)
            )
    return pk


def _ideal_tree_to_root(
    rows: int, cols: int, chunk: int, payload: str, col_color: int, row_color: int
) -> list[dict]:
    """Reverse spanning tree to corner (0,0): columns converge north, then row 0 west."""
    pk: list[dict] = []
    for c in range(cols):
        for r in range(rows - 1, 0, -1):
            level = (rows - 1) - r
            pk.append(_cedge(r * cols + c, (r - 1) * cols + c, chunk, payload, level, col_color))
    base = max(0, rows - 1)
    for c in range(cols - 1, 0, -1):
        delay = base + (cols - 1 - c)
        pk.append(_cedge(c, c - 1, chunk, payload, delay, row_color))
    return pk


def _ideal_allgather(rows: int, cols: int, chunk: int) -> list[dict]:
    """2D ring allgather: bidirectional row pass, then bidirectional column pass."""
    pk: list[dict] = []
    for step in range(cols - 1):
        for r in range(rows):
            for c in range(cols - 1):
                pk.append(_cedge(r * cols + c, r * cols + c + 1, chunk, "allgather", step, C_AG_ROW_EAST))
            for c in range(cols - 1, 0, -1):
                pk.append(_cedge(r * cols + c, r * cols + c - 1, chunk, "allgather", step, C_AG_ROW_WEST))
    base = max(0, cols - 1)
    for step in range(rows - 1):
        for c in range(cols):
            for r in range(rows - 1):
                pk.append(
                    _cedge(r * cols + c, (r + 1) * cols + c, chunk, "allgather", base + step, C_AG_COL_SOUTH)
                )
            for r in range(rows - 1, 0, -1):
                pk.append(
                    _cedge(r * cols + c, (r - 1) * cols + c, chunk, "allgather", base + step, C_AG_COL_NORTH)
                )
    return pk


def _ideal_allreduce(rows: int, cols: int, chunk: int) -> list[dict]:
    """2D reduce-scatter + allgather along rows then columns."""
    pk: list[dict] = []
    t = 0
    for step in range(cols - 1):
        for r in range(rows):
            for c in range(cols - 1):
                pk.append(_cedge(r * cols + c, r * cols + c + 1, chunk, "allreduce", t + step, C_AR_RS_ROW_EAST))
    t += max(1, cols - 1)
    for step in range(cols - 1):
        for r in range(rows):
            for c in range(cols - 1, 0, -1):
                pk.append(_cedge(r * cols + c, r * cols + c - 1, chunk, "allreduce", t + step, C_AR_AG_ROW_WEST))
    t += max(1, cols - 1)
    for step in range(rows - 1):
        for c in range(cols):
            for r in range(rows - 1):
                pk.append(_cedge(r * cols + c, (r + 1) * cols + c, chunk, "allreduce", t + step, C_AR_RS_COL_SOUTH))
    t += max(1, rows - 1)
    for step in range(rows - 1):
        for c in range(cols):
            for r in range(rows - 1, 0, -1):
                pk.append(_cedge(r * cols + c, (r - 1) * cols + c, chunk, "allreduce", t + step, C_AR_AG_COL_NORTH))
    return pk

"""MoE decode FFN workload generation."""

from __future__ import annotations

from dataclasses import dataclass

from wsesim.workload.ops import GEMMOp, LLMWorkload


def generate_moe_decode_ffn_workload(
    model_name: str,
    hidden_dim: int,
    expert_ffn_dim: int,
    num_experts: int,
    top_k: int,
    decode_tokens: int,
) -> LLMWorkload:
    ops: list[GEMMOp] = []
    gate = GEMMOp(
        name="router_gate_proj",
        m=decode_tokens,
        n=num_experts,
        k=hidden_dim,
        depends_on=[],
        output_to=["token_dispatch"],
    )
    ops.append(gate)

    dispatch = GEMMOp(
        name="token_dispatch",
        m=decode_tokens,
        n=top_k,
        k=num_experts,
        depends_on=[gate.name],
        output_to=[],
    )
    ops.append(dispatch)

    for expert_idx in range(num_experts):
        gate_proj = GEMMOp(
            name=f"expert_{expert_idx}_gate_proj",
            m=decode_tokens,
            n=expert_ffn_dim,
            k=hidden_dim,
            depends_on=[dispatch.name],
            output_to=[f"expert_{expert_idx}_down_proj"],
        )
        down_proj = GEMMOp(
            name=f"expert_{expert_idx}_down_proj",
            m=decode_tokens,
            n=hidden_dim,
            k=expert_ffn_dim,
            depends_on=[gate_proj.name],
            output_to=["token_combine"],
        )
        ops.extend([gate_proj, down_proj])

    combine = GEMMOp(
        name="token_combine",
        m=decode_tokens,
        n=hidden_dim,
        k=top_k,
        depends_on=[f"expert_{idx}_down_proj" for idx in range(num_experts)],
        output_to=[],
    )
    ops.append(combine)

    return LLMWorkload(
        model_name=model_name,
        ops=ops,
        metadata={
            "hidden_dim": hidden_dim,
            "expert_ffn_dim": expert_ffn_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "decode_tokens": decode_tokens,
        },
    )


@dataclass(slots=True)
class DeepSeekV3FFNProfile:
    hidden_dim: int = 7168
    expert_ffn_dim: int = 18432
    num_routed_experts: int = 256
    num_shared_experts: int = 1
    top_k: int = 8
    decode_tokens: int = 32
    routing_skew_alpha: float = 1.2
    capacity_factor: float = 1.25


def generate_deepseek_v3_decode_ffn_workload(
    profile: DeepSeekV3FFNProfile,
) -> LLMWorkload:
    """Generate a DeepSeek-V3-like MoE FFN decode workload graph.

    The model captures routed experts + shared experts and a skewed top-k token
    routing distribution to make per-expert compute more realistic on WSE.
    """
    ops: list[GEMMOp] = []
    gate = GEMMOp(
        name="deepseek_v3_router_gate_proj",
        m=profile.decode_tokens,
        n=profile.num_routed_experts,
        k=profile.hidden_dim,
        op_type="router",
        depends_on=[],
        output_to=["deepseek_v3_token_dispatch"],
    )
    dispatch = GEMMOp(
        name="deepseek_v3_token_dispatch",
        m=profile.decode_tokens,
        n=profile.top_k,
        k=profile.num_routed_experts,
        op_type="dispatch",
        depends_on=[gate.name],
        output_to=[],
    )
    ops.extend([gate, dispatch])

    routed_token_loads = _estimate_routed_expert_token_loads(
        num_experts=profile.num_routed_experts,
        decode_tokens=profile.decode_tokens,
        top_k=profile.top_k,
        skew_alpha=profile.routing_skew_alpha,
        capacity_factor=profile.capacity_factor,
    )

    routed_down_ops: list[str] = []
    active_routed_experts = 0
    for expert_id, token_load in enumerate(routed_token_loads):
        if token_load <= 0:
            continue
        active_routed_experts += 1
        up = GEMMOp(
            name=f"deepseek_v3_expert_routed_{expert_id}_up_proj",
            m=token_load,
            n=profile.expert_ffn_dim,
            k=profile.hidden_dim,
            op_type="expert_up_proj",
            expert_id=expert_id,
            expert_kind="routed",
            depends_on=[dispatch.name],
            output_to=[f"deepseek_v3_expert_routed_{expert_id}_down_proj"],
        )
        down = GEMMOp(
            name=f"deepseek_v3_expert_routed_{expert_id}_down_proj",
            m=token_load,
            n=profile.hidden_dim,
            k=profile.expert_ffn_dim,
            op_type="expert_down_proj",
            expert_id=expert_id,
            expert_kind="routed",
            depends_on=[up.name],
            output_to=["deepseek_v3_token_combine"],
        )
        ops.extend([up, down])
        routed_down_ops.append(down.name)

    shared_down_ops: list[str] = []
    for shared_idx in range(profile.num_shared_experts):
        up = GEMMOp(
            name=f"deepseek_v3_expert_shared_{shared_idx}_up_proj",
            m=profile.decode_tokens,
            n=profile.expert_ffn_dim,
            k=profile.hidden_dim,
            op_type="expert_up_proj",
            expert_id=shared_idx,
            expert_kind="shared",
            depends_on=[dispatch.name],
            output_to=[f"deepseek_v3_expert_shared_{shared_idx}_down_proj"],
        )
        down = GEMMOp(
            name=f"deepseek_v3_expert_shared_{shared_idx}_down_proj",
            m=profile.decode_tokens,
            n=profile.hidden_dim,
            k=profile.expert_ffn_dim,
            op_type="expert_down_proj",
            expert_id=shared_idx,
            expert_kind="shared",
            depends_on=[up.name],
            output_to=["deepseek_v3_token_combine"],
        )
        ops.extend([up, down])
        shared_down_ops.append(down.name)

    combine = GEMMOp(
        name="deepseek_v3_token_combine",
        m=profile.decode_tokens,
        n=profile.hidden_dim,
        k=profile.top_k + profile.num_shared_experts,
        op_type="combine",
        depends_on=routed_down_ops + shared_down_ops,
        output_to=[],
    )
    ops.append(combine)

    return LLMWorkload(
        model_name="deepseek_v3_ffn_decode",
        ops=ops,
        metadata={
            "hidden_dim": profile.hidden_dim,
            "expert_ffn_dim": profile.expert_ffn_dim,
            "num_routed_experts": profile.num_routed_experts,
            "num_shared_experts": profile.num_shared_experts,
            "top_k": profile.top_k,
            "decode_tokens": profile.decode_tokens,
            "routing_skew_alpha": profile.routing_skew_alpha,
            "capacity_factor": profile.capacity_factor,
            "active_routed_experts": active_routed_experts,
        },
    )


def _estimate_routed_expert_token_loads(
    num_experts: int,
    decode_tokens: int,
    top_k: int,
    skew_alpha: float,
    capacity_factor: float,
) -> list[int]:
    total_assignments = decode_tokens * top_k
    if total_assignments <= 0 or num_experts <= 0:
        return [0 for _ in range(max(num_experts, 0))]

    weights = [1.0 / ((idx + 1) ** max(skew_alpha, 0.01)) for idx in range(num_experts)]
    weight_sum = sum(weights)
    normalized = [w / weight_sum for w in weights]
    hard_cap = max(1, int(capacity_factor * total_assignments / num_experts))

    loads = [min(hard_cap, int(round(p * total_assignments))) for p in normalized]
    assigned = sum(loads)
    idx = 0
    while assigned < total_assignments:
        if loads[idx % num_experts] < hard_cap:
            loads[idx % num_experts] += 1
            assigned += 1
        idx += 1
        if idx > total_assignments * 4:
            break
    idx = 0
    while assigned > total_assignments:
        pos = idx % num_experts
        if loads[pos] > 0:
            loads[pos] -= 1
            assigned -= 1
        idx += 1
        if idx > total_assignments * 4:
            break
    return loads

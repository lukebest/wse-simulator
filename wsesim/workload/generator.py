"""MoE decode FFN workload generation."""

from __future__ import annotations

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

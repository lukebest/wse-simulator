"""MoE decode FFN workload generation."""

from __future__ import annotations

import random

from wsesim.workload.ops import GEMMOp, LLMWorkload, TokenRoute


def generate_moe_decode_ffn_workload(
    model_name: str,
    hidden_dim: int,
    expert_ffn_dim: int,
    num_experts: int,
    top_k: int,
    decode_tokens: int,
    seed: int = 1234,
) -> LLMWorkload:
    token_routes = _generate_token_routes(
        decode_tokens=decode_tokens,
        num_experts=num_experts,
        top_k=top_k,
        seed=seed,
    )
    expert_token_counts = _expert_token_counts(token_routes, num_experts)

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
        expert_tokens = expert_token_counts[expert_idx]
        gate_proj = GEMMOp(
            name=f"expert_{expert_idx}_gate_proj",
            m=expert_tokens,
            n=expert_ffn_dim,
            k=hidden_dim,
            depends_on=[dispatch.name],
            output_to=[f"expert_{expert_idx}_down_proj"],
        )
        down_proj = GEMMOp(
            name=f"expert_{expert_idx}_down_proj",
            m=expert_tokens,
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
        token_routes=token_routes,
        metadata={
            "hidden_dim": hidden_dim,
            "expert_ffn_dim": expert_ffn_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "decode_tokens": decode_tokens,
            "seed": seed,
        },
    )


def _generate_token_routes(
    decode_tokens: int,
    num_experts: int,
    top_k: int,
    seed: int,
) -> list[TokenRoute]:
    rng = random.Random(seed)
    routes: list[TokenRoute] = []
    for token_id in range(decode_tokens):
        experts = rng.sample(range(num_experts), k=min(top_k, num_experts))
        raw_scores = [rng.random() for _ in experts]
        score_sum = sum(raw_scores) or 1.0
        norm_scores = [score / score_sum for score in raw_scores]
        routes.append(
            TokenRoute(
                token_id=token_id,
                selected_experts=experts,
                gate_scores=norm_scores,
            )
        )
    return routes


def _expert_token_counts(token_routes: list[TokenRoute], num_experts: int) -> list[int]:
    counts = [0 for _ in range(num_experts)]
    for route in token_routes:
        for expert_id in route.selected_experts:
            counts[expert_id] += 1
    return counts

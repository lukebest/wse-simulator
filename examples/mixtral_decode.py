"""Generate and inspect MoE decode FFN workload."""

from __future__ import annotations

from wsesim.workload.generator import generate_moe_decode_ffn_workload


def main() -> None:
    workload = generate_moe_decode_ffn_workload(
        model_name="mixtral_8x7b",
        hidden_dim=4096,
        expert_ffn_dim=14336,
        num_experts=8,
        top_k=2,
        decode_tokens=32,
    )
    print("model:", workload.model_name)
    print("num_ops:", len(workload.ops))
    print("first_op:", workload.ops[0].name)
    print("last_op:", workload.ops[-1].name)


if __name__ == "__main__":
    main()

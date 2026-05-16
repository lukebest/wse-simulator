# WSE Simulator

`wsesim` is a modular Wafer Scale Engine simulator focused on:

- Wafer -> Reticle -> Core hierarchy
- Unified NoC/NoW network modeling
- Fault-aware mapping and routing
- MoE LLM decode FFN workload mapping
- Design-space exploration (DSE)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## DSE + Pareto Workflow

```bash
source .venv/bin/activate
python examples/run_dse.py
python examples/plot_pareto.py --trials-csv outputs/dse_trials.csv --pareto-csv outputs/dse_pareto.csv --output-dir outputs
```

This generates:

- `outputs/dse_trials.json`
- `outputs/dse_trials.csv`
- `outputs/dse_pareto.csv`
- `outputs/pareto_latency_vs_throughput.png`
- `outputs/pareto_latency_vs_congestion.png`

`examples/run_dse.py` now evaluates configurations using a DeepSeek-V3-style
decode FFN workload model (routed experts + shared experts + top-k routing).
The evaluator uses hierarchical communication simulation: NoC within reticles
and NoW across reticles.
Gateway modeling is configurable via `network.gateways_per_reticle` and
`network.gateway_policy`.

## DeepSeek-V3 FFN Mapping Example

```bash
source .venv/bin/activate
python examples/deepseek_v3_mapping.py
```

The DeepSeek-V3 profile is provided in `configs/deepseek_v3_ffn.yaml`.
It models:

- Routed experts + shared experts
- Top-k decode routing
- Skewed expert token load distribution
- Expert-affinity mapping onto WSE cores

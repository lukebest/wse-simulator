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

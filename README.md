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

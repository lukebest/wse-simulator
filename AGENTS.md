## Cursor Cloud specific instructions

**Product:** `wsesim` — a pure-Python Wafer Scale Engine simulator (NoC/NoW networks, fault-aware mapping, MoE LLM workloads, DSE). No external services or databases required.

**Standard commands** (see `README.md` and `pyproject.toml`):

- **Lint:** `ruff check .`
- **Tests:** `pytest` (50 tests, ~26s)
- **Example scripts:** `python3 examples/deepseek_v4_pro_mapping.py`, `python3 examples/run_dse.py`

**Caveats:**

- Use `python3` not `python` — the system does not alias `python` to `python3`.
- The `[dse]` extra (`scikit-optimize`) is needed for DSE-related tests and examples; install with `pip install -e ".[dev,dse]"`.
- `ruff check .` exits non-zero due to 4 pre-existing lint warnings (unused variables / imports); these are in the existing codebase, not regressions.
- The `outputs/` directory is created by DSE example scripts; it is gitignored.

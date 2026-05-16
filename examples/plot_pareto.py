"""Plot DSE trial and Pareto front scatter charts from CSV exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from wsesim.dse.plot import plot_pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Pareto analysis from DSE CSV exports.")
    parser.add_argument(
        "--trials-csv",
        type=Path,
        default=Path("outputs/dse_trials.csv"),
        help="Path to exported trials CSV.",
    )
    parser.add_argument(
        "--pareto-csv",
        type=Path,
        default=Path("outputs/dse_pareto.csv"),
        help="Path to exported Pareto CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write plot images.",
    )
    args = parser.parse_args()

    out1, out2 = plot_pareto(args.trials_csv, args.pareto_csv, args.output_dir)
    print("saved:", out1)
    print("saved:", out2)


if __name__ == "__main__":
    main()

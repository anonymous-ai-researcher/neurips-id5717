#!/usr/bin/env python3
"""Generate LaTeX tables and TikZ figures from experiment results."""

import argparse
import json
import numpy as np
from pathlib import Path


def generate_table_qual(summary):
    """Generate Table 1: Certified qualitative results."""
    rq1 = summary.get("rq1", {})
    n = rq1.get("n_instances", 540)

    print(r"\begin{table}[t]")
    print(r"\caption{Certified qualitative reasoning on "
          f"{n} UNSAT instances.}}")
    print(r"\centering\small")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"& CaDiCaL & RoundingSat & CMS-TNN \\")
    print(r"\midrule")
    print(f"Coverage & {rq1.get('cadical_coverage', 'N/A')} "
          f"({100*rq1.get('cadical_coverage',0)/n:.0f}\\%) "
          f"& {rq1.get('roundingsat_coverage', 'N/A')} "
          f"({100*rq1.get('roundingsat_coverage',0)/n:.0f}\\%) "
          f"& \\textbf{{{rq1.get('cms_coverage', 'N/A')}}} "
          f"(\\textbf{{{100*rq1.get('cms_coverage',0)/n:.0f}\\%}}) \\\\")
    print(f"PAR-2 (s) & {rq1.get('cadical_par2', 'N/A'):.0f} "
          f"& {rq1.get('roundingsat_par2', 'N/A'):.0f} "
          f"& \\textbf{{{rq1.get('cms_par2', 'N/A'):.0f}}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--table", type=str, choices=["qual", "quant",
                        "ablation"])
    parser.add_argument("--figure", type=str, choices=["certsize", "scaling"])
    parser.add_argument("--output", type=str, default="tables/")
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)

    if args.table == "qual":
        generate_table_qual(summary)
    else:
        print(f"Table/figure '{args.table or args.figure}' "
              "generation not yet implemented.")
        print("Use the raw data in the summary JSON to generate manually.")


if __name__ == "__main__":
    main()

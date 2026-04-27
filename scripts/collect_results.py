#!/usr/bin/env python3
"""Collect experiment results from log files and compute summary statistics.

Parses solver/checker/counter logs, computes PAR-2, coverage, certificate sizes,
and runs Wilcoxon signed-rank tests for all pairwise comparisons.
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats


QUAL_TIMEOUT = 500
QUANT_TIMEOUT = 5000


def parse_solver_log(log_path):
    """Extract timing, result, and stats from a solver log file."""
    result = {"status": "timeout", "time": None, "cert_size": None,
              "props": None}
    if not os.path.exists(log_path):
        return result

    with open(log_path) as f:
        content = f.read()

    if "s UNSATISFIABLE" in content:
        result["status"] = "unsat"
    elif "s SATISFIABLE" in content:
        result["status"] = "sat"

    # Wall clock time from /usr/bin/time
    m = re.search(r"Elapsed \(wall clock\) time.*?(\d+):(\d+\.\d+)", content)
    if m:
        result["time"] = int(m.group(1)) * 60 + float(m.group(2))

    # Certificate size
    m = re.search(r"proof.*?(\d+(?:\.\d+)?)\s*(?:MB|bytes)", content)
    if m:
        result["cert_size"] = float(m.group(1))

    # Propagation count
    m = re.search(r"props\s*:\s*(\d+)", content)
    if m:
        result["props"] = int(m.group(1))

    return result


def compute_par2(times, timeout, total):
    """Compute PAR-2: average runtime with 2x penalty for unsolved."""
    solved = [t for t in times if t is not None and t <= timeout]
    unsolved = total - len(solved)
    return (sum(solved) + 2 * timeout * unsolved) / total


def wilcoxon_test(times_a, times_b, name_a, name_b):
    """Run Wilcoxon signed-rank test on paired runtimes."""
    # Filter pairs where both have valid times
    pairs = [(a, b) for a, b in zip(times_a, times_b)
             if a is not None and b is not None]
    if len(pairs) < 10:
        return {"comparison": f"{name_a} vs {name_b}", "n": len(pairs),
                "p_value": None, "note": "insufficient data"}

    a_vals, b_vals = zip(*pairs)
    stat, p_value = stats.wilcoxon(a_vals, b_vals, alternative="two-sided")
    r = 1 - (2 * stat) / (len(pairs) * (len(pairs) + 1))

    return {
        "comparison": f"{name_a} vs {name_b}",
        "n": len(pairs),
        "W": float(stat),
        "p_value": float(p_value),
        "r_effect_size": float(r),
        "conclusion": f"{name_a} faster" if np.mean(a_vals) < np.mean(b_vals)
                      else f"{name_b} faster",
    }


def collect_rq1(results_dir):
    """Collect RQ1 (certified qualitative) results."""
    rq1_dir = Path(results_dir) / "rq1"
    if not rq1_dir.exists():
        return None

    results = defaultdict(list)
    for log in sorted(rq1_dir.glob("*_cms.log")):
        prefix = str(log).replace("_cms.log", "")
        results["cms_solve"].append(parse_solver_log(f"{prefix}_cms.log"))
        results["cms_elab"].append(parse_solver_log(f"{prefix}_elab.log"))
        results["cms_check"].append(parse_solver_log(f"{prefix}_check.log"))
        results["cadical"].append(parse_solver_log(f"{prefix}_cadical.log"))
        results["cakelpr"].append(parse_solver_log(f"{prefix}_cakelpr.log"))
        results["roundingsat"].append(
            parse_solver_log(f"{prefix}_roundingsat.log"))
        results["veripb"].append(parse_solver_log(f"{prefix}_veripb.log"))

    n = len(results["cms_solve"])
    if n == 0:
        return None

    # Compute certified total times
    cms_times = []
    for s, e, c in zip(results["cms_solve"], results["cms_elab"],
                        results["cms_check"]):
        if all(x["time"] is not None for x in [s, e, c]):
            cms_times.append(s["time"] + e["time"] + c["time"])
        else:
            cms_times.append(None)

    cadical_times = []
    for s, c in zip(results["cadical"], results["cakelpr"]):
        if all(x["time"] is not None for x in [s, c]):
            cadical_times.append(s["time"] + c["time"])
        else:
            cadical_times.append(None)

    rs_times = []
    for s, c in zip(results["roundingsat"], results["veripb"]):
        if all(x["time"] is not None for x in [s, c]):
            rs_times.append(s["time"] + c["time"])
        else:
            rs_times.append(None)

    summary = {
        "n_instances": n,
        "cms_coverage": sum(1 for t in cms_times if t is not None
                           and t <= QUAL_TIMEOUT),
        "cms_par2": compute_par2(cms_times, QUAL_TIMEOUT, n),
        "cadical_coverage": sum(1 for t in cadical_times if t is not None
                               and t <= QUAL_TIMEOUT),
        "cadical_par2": compute_par2(cadical_times, QUAL_TIMEOUT, n),
        "roundingsat_coverage": sum(1 for t in rs_times if t is not None
                                   and t <= QUAL_TIMEOUT),
        "roundingsat_par2": compute_par2(rs_times, QUAL_TIMEOUT, n),
        "wilcoxon_cms_vs_cadical": wilcoxon_test(
            cms_times, cadical_times, "CMS-TNN", "CaDiCaL"),
        "wilcoxon_cms_vs_roundingsat": wilcoxon_test(
            cms_times, rs_times, "CMS-TNN", "RoundingSat"),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Collect experiment results")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/summary.json")
    args = parser.parse_args()

    summary = {}

    rq1 = collect_rq1(args.results)
    if rq1:
        summary["rq1"] = rq1
        print(f"RQ1: {rq1['cms_coverage']}/{rq1['n_instances']} certified "
              f"({100*rq1['cms_coverage']/rq1['n_instances']:.0f}%), "
              f"PAR-2={rq1['cms_par2']:.0f}s")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {args.output}")


if __name__ == "__main__":
    main()

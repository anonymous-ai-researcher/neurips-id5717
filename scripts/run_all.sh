#!/bin/bash
# Run all experiments for the certified TNN verification paper.
# Usage: bash scripts/run_all.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Tool paths
CMS_TNN="${ROOT_DIR}/src/solver/build/cms-tnn"
CMS_CNF="${ROOT_DIR}/src/solver/build/cms-cnf"  # ablation: CNF pre-encoding
ELABORATOR="${ROOT_DIR}/src/elaborator/frat-xor-tnn"
CHECKER="${ROOT_DIR}/src/checker/cake_xlrup_tnn"
COUNTER="${ROOT_DIR}/src/counter/build/approxmc-tnn"
CERTCHECK="${ROOT_DIR}/src/certcheck/build/certcheck-tnn"

# Baseline tools
CADICAL="${ROOT_DIR}/baselines/cadical"
CAKE_LPR="${ROOT_DIR}/baselines/cake_lpr"
ROUNDINGSAT="${ROOT_DIR}/baselines/roundingsat"
VERIPB="${ROOT_DIR}/baselines/veripb"
APPROXMC_CNF="${ROOT_DIR}/baselines/approxmccert-cnf"
CERTCHECK_CNF="${ROOT_DIR}/baselines/certcheck-cnf"

# Configuration
BENCHMARK_DIR="${ROOT_DIR}/benchmarks/instances"
RESULTS_DIR="${ROOT_DIR}/results"
QUAL_TIMEOUT=500       # seconds
QUANT_TIMEOUT=5000     # seconds
MEMORY_LIMIT=16384     # MB (16 GB)
NUM_SEEDS=5

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "${RESULTS_DIR}"/{rq1,rq2,rq3,rq4,rq5}

# ─── Utility functions ──────────────────────────────────────────────────────

run_with_limits() {
    local timeout=$1; shift
    local memlimit=$1; shift
    local logfile=$1; shift

    if $DRY_RUN; then
        echo "[DRY RUN] timeout=${timeout}s mem=${memlimit}MB: $*" | tee -a "$logfile"
        return 0
    fi

    /usr/bin/time -v timeout --signal=KILL "${timeout}s" \
        bash -c "ulimit -v $((memlimit * 1024)); $*" \
        > "$logfile" 2>&1 || true
}

get_instances() {
    local type=$1  # "unsat" or "all"
    if [[ "$type" == "unsat" ]]; then
        find "$BENCHMARK_DIR" -name "unsat_*.cnf-tnn" | sort
    else
        find "$BENCHMARK_DIR" -name "*.cnf-tnn" | sort
    fi
}

# ─── RQ1: Certified Qualitative Reasoning (540 UNSAT instances) ─────────

run_rq1() {
    echo "=== RQ1: Certified Qualitative Reasoning ==="
    local rq1_dir="${RESULTS_DIR}/rq1"

    while IFS= read -r instance; do
        local name
        name=$(basename "$instance" .cnf-tnn)
        local net_dir
        net_dir=$(basename "$(dirname "$(dirname "$instance")")")
        local eps_dir
        eps_dir=$(basename "$(dirname "$instance")")
        local prefix="${rq1_dir}/${net_dir}_${eps_dir}_${name}"

        # --- CMS-TNN + cake_xlrup-TNN ---
        echo "  [CMS+cake] $net_dir/$eps_dir/$name"
        run_with_limits "$QUAL_TIMEOUT" "$MEMORY_LIMIT" \
            "${prefix}_cms.log" \
            "$CMS_TNN --proof ${prefix}_cms.frat $instance"

        # Elaborate FRAT -> XLRUP
        if [[ -f "${prefix}_cms.frat" ]]; then
            run_with_limits "$QUAL_TIMEOUT" "$MEMORY_LIMIT" \
                "${prefix}_elab.log" \
                "$ELABORATOR ${prefix}_cms.frat ${prefix}_cms.xlrup $instance"
        fi

        # Check with cake_xlrup-TNN
        if [[ -f "${prefix}_cms.xlrup" ]]; then
            run_with_limits "$QUAL_TIMEOUT" "$MEMORY_LIMIT" \
                "${prefix}_check.log" \
                "$CHECKER $instance ${prefix}_cms.xlrup"
        fi

        # --- CaDiCaL + cake_lpr ---
        echo "  [CaDiCaL+cake_lpr] $net_dir/$eps_dir/$name"
        # First encode to CNF
        local cnf_file="${prefix}_cadical.cnf"
        python3 "${SCRIPT_DIR}/encode_cnf.py" "$instance" "$cnf_file" 2>/dev/null || true

        if [[ -f "$cnf_file" ]]; then
            run_with_limits "$QUAL_TIMEOUT" "$MEMORY_LIMIT" \
                "${prefix}_cadical.log" \
                "$CADICAL --proof ${prefix}_cadical.lrat $cnf_file"

            if [[ -f "${prefix}_cadical.lrat" ]]; then
                run_with_limits "$QUAL_TIMEOUT" "$MEMORY_LIMIT" \
                    "${prefix}_cakelpr.log" \
                    "$CAKE_LPR $cnf_file ${prefix}_cadical.lrat"
            fi
        fi

        # --- RoundingSat + VeriPB ---
        echo "  [RoundingSat+VeriPB] $net_dir/$eps_dir/$name"
        local opb_file="${prefix}_roundingsat.opb"
        python3 "${SCRIPT_DIR}/encode_opb.py" "$instance" "$opb_file" 2>/dev/null || true

        if [[ -f "$opb_file" ]]; then
            run_with_limits "$QUAL_TIMEOUT" "$MEMORY_LIMIT" \
                "${prefix}_roundingsat.log" \
                "$ROUNDINGSAT --proof ${prefix}_roundingsat.pbp $opb_file"

            if [[ -f "${prefix}_roundingsat.pbp" ]]; then
                run_with_limits "$QUAL_TIMEOUT" "$MEMORY_LIMIT" \
                    "${prefix}_veripb.log" \
                    "$VERIPB $opb_file ${prefix}_roundingsat.pbp"
            fi
        fi

    done < <(get_instances "unsat")
}

# ─── RQ2: Certified Quantitative Reasoning (1,080 instances) ────────────

run_rq2() {
    echo "=== RQ2: Certified Quantitative Reasoning ==="
    local rq2_dir="${RESULTS_DIR}/rq2"

    while IFS= read -r instance; do
        local name
        name=$(basename "$instance" .cnf-tnn)
        local net_dir
        net_dir=$(basename "$(dirname "$(dirname "$instance")")")
        local eps_dir
        eps_dir=$(basename "$(dirname "$instance")")
        local prefix="${rq2_dir}/${net_dir}_${eps_dir}_${name}"

        # --- ApproxMCCert-TNN + CertCheck-TNN ---
        echo "  [ApproxMC-TNN] $net_dir/$eps_dir/$name"
        run_with_limits "$QUANT_TIMEOUT" "$MEMORY_LIMIT" \
            "${prefix}_approxmc.log" \
            "$COUNTER --eta 0.8 --delta 0.2 --cert ${prefix}_approxmc.cert $instance"

        if [[ -f "${prefix}_approxmc.cert" ]]; then
            run_with_limits "$QUANT_TIMEOUT" "$MEMORY_LIMIT" \
                "${prefix}_certcheck.log" \
                "$CERTCHECK $instance ${prefix}_approxmc.cert"
        fi

        # --- ApproxMCCert (CNF baseline) ---
        echo "  [ApproxMC-CNF] $net_dir/$eps_dir/$name"
        local cnf_file="${prefix}_cnf.cnf"
        python3 "${SCRIPT_DIR}/encode_cnf.py" "$instance" "$cnf_file" 2>/dev/null || true

        if [[ -f "$cnf_file" ]]; then
            run_with_limits "$QUANT_TIMEOUT" "$MEMORY_LIMIT" \
                "${prefix}_approxmc_cnf.log" \
                "$APPROXMC_CNF --eta 0.8 --delta 0.2 --cert ${prefix}_cnf.cert $cnf_file"

            if [[ -f "${prefix}_cnf.cert" ]]; then
                run_with_limits "$QUANT_TIMEOUT" "$MEMORY_LIMIT" \
                    "${prefix}_certcheck_cnf.log" \
                    "$CERTCHECK_CNF $cnf_file ${prefix}_cnf.cert"
            fi
        fi

    done < <(get_instances "all")
}

# ─── RQ3: Ablation (540 UNSAT instances) ────────────────────────────────

run_rq3() {
    echo "=== RQ3: Ablation on Native CTCC Propagation ==="
    local rq3_dir="${RESULTS_DIR}/rq3"

    while IFS= read -r instance; do
        local name
        name=$(basename "$instance" .cnf-tnn)
        local net_dir
        net_dir=$(basename "$(dirname "$(dirname "$instance")")")
        local eps_dir
        eps_dir=$(basename "$(dirname "$instance")")
        local prefix="${rq3_dir}/${net_dir}_${eps_dir}_${name}"

        # CMS-CNF (pre-encodes CTCCs to CNF)
        echo "  [CMS-CNF] $net_dir/$eps_dir/$name"
        run_with_limits "$QUAL_TIMEOUT" "$MEMORY_LIMIT" \
            "${prefix}_cmscnf.log" \
            "$CMS_CNF --proof ${prefix}_cmscnf.frat --stats $instance"

    done < <(get_instances "unsat")
}

# ─── Main ────────────────────────────────────────────────────────────────

echo "Certified TNN Verification Experiments"
echo "======================================="
echo "Benchmark: ${BENCHMARK_DIR}"
echo "Results:   ${RESULTS_DIR}"
echo "Timeouts:  qual=${QUAL_TIMEOUT}s, quant=${QUANT_TIMEOUT}s"
echo "Memory:    ${MEMORY_LIMIT} MB"
echo ""

run_rq1
run_rq2
run_rq3

echo ""
echo "=== All experiments completed ==="
echo "Run: python scripts/collect_results.py --results ${RESULTS_DIR}"

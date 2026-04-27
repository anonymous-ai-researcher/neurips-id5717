#!/bin/bash
# Run RQ experiments. See run_all.sh for full details.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Running RQ3: Ablation on Native CTCC Propagation (540 UNSAT instances)"
source "${SCRIPT_DIR}/run_all.sh" && run_rq3

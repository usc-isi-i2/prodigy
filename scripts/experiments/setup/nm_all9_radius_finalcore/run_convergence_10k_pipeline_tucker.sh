#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[pipeline] 10k seed-0 training starts utc=$(date -u +%FT%TZ)"
bash "$SCRIPT_DIR/run_convergence_10k_tucker.sh"
echo "[pipeline] training complete; four-panel evaluation starts utc=$(date -u +%FT%TZ)"
bash "$SCRIPT_DIR/run_convergence_10k_evaluation_tucker.sh"
echo "[pipeline] complete utc=$(date -u +%FT%TZ)"

#!/usr/bin/env bash
set -euo pipefail

# Queue a no-clobber recovery from the dedicated recovery worktree. The failed
# run directory and log must be moved to a timestamped archive before launch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ORIGINAL_SESSION="${ORIGINAL_SESSION:-archmatrix_full_100}"

while tmux has-session -t "$ORIGINAL_SESSION" 2>/dev/null; do
  sleep 30
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
export STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-archmatrix/state/icl_arch_matrix}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_matrix_recovery}"
export GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
export POLL_SECONDS="${POLL_SECONDS:-30}"

bash "$SCRIPT_DIR/run_matrix_tucker.sh"

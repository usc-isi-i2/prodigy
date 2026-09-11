#!/usr/bin/env bash
set -euo pipefail
export STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/state/node_mlp_ladder_s0_2500}"
export RESULTS_ROOT="${RESULTS_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/results/node_mlp_ladder_s0_2500/raw}"
export LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/log/node_mlp_ladder_s0_2500}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_fast_mlp_ladder_tucker.sh" "$@"

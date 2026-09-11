#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FANOUTS=(1 10 50 100)
for fanout in "${FANOUTS[@]}"; do
  state="$ROOT/state/context_mlp_fanout/f${fanout}"
  results="$ROOT/results/context_mlp_fanout/f${fanout}"
  for view in neighborhood node_neighborhood; do
    for objective in fp lp; do
      echo "fanout=$fanout view=$view objective=$objective"
      if [[ "${DRY_RUN:-0}" == 1 ]]; then continue; fi
      FANOUT="$fanout" STATE_ROOT="$state" LOG_ROOT="$ROOT/log/context_mlp_fanout/f${fanout}/train_${view}_${objective}" \
        bash scripts/run_context_mlp_transfer_tucker.sh "$view" "$objective" full
      FANOUT="$fanout" STATE_ROOT="$state" RESULTS_ROOT="$results" LOG_ROOT="$ROOT/log/context_mlp_fanout/f${fanout}/eval_${view}_${objective}" \
        bash scripts/eval_context_mlp_transfer_tucker.sh "$view" "$objective"
    done
  done
done

#!/usr/bin/env bash
# Read-only resource gate; run from the dedicated Tucker worktree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
GRAPH="${DATA_ROOT:-/dataMeR1/phil/data}/merged/graphs/ukr_rus_covid_midterm_all8_retweet_graph.pt"
[[ -f "${GRAPH}" ]] || { echo "missing all8 graph: ${GRAPH}" >&2; exit 1; }
[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == "codex/nm-ladder-sequential-nhop2" ]] || {
  echo "wrong branch in ${REPO_ROOT}" >&2
  exit 1
}
python3 "${REPO_ROOT}/scripts/experiments/setup/nm_ladder_sequential_nhop2/make_configs.py" --check
echo "inputs OK: ${GRAPH}"

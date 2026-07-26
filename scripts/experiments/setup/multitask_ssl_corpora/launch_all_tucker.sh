#!/usr/bin/env bash
# Launch all 8 multitask_ssl_corpora pretrains (2 corpora x 4 arms) as detached
# tmux sessions on Tucker, 2 runs per GPU, corpora paired per arm so the long
# poles overlap (NM is the slow arm; CL/FP are fast):
#   GPU0: cov_NM  + all8_NM      GPU1: cov_MIX + all8_MIX
#   GPU2: cov_CL  + all8_CL      GPU3: cov_FP  + all8_FP   (lightest pair -> the
#                                                            partially-occupied GPU)
# Uses the env-python-direct pattern (NOT conda activate) because conda is not
# initialized in detached non-interactive tmux (see AGENTS.md / the rotation run).
# Logs: /tmp/msc_<corpus>_<ARM>.log ; state: <repo>/state/msc_<corpus>_<ARM>_<ts>/.
#
# Usage (from the Tucker worktree holding this branch):
#   bash scripts/experiments/setup/multitask_ssl_corpora/launch_all_tucker.sh
#   ONLY="cov_NM all8_MIX" bash .../launch_all_tucker.sh   # relaunch a subset
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # setup/<name> is 4 levels below repo root
PY="${PY:-/home/mhchu/miniconda3/envs/prodigy/bin/python}"
ENV_LIB="${ENV_LIB:-/home/mhchu/miniconda3/envs/prodigy/lib}"

# run -> GPU map (edit here if the GPU landscape changes; GPUs 0-3 ONLY)
declare -A GPU=(
  [cov_NM]=0  [all8_NM]=0
  [cov_MIX]=1 [all8_MIX]=1
  [cov_CL]=2  [all8_CL]=2
  [cov_FP]=3  [all8_FP]=3
)
RUNS="${ONLY:-cov_NM all8_NM cov_MIX all8_MIX cov_CL all8_CL cov_FP all8_FP}"

for run in ${RUNS}; do
  corpus="${run%%_*}"; arm="${run#*_}"
  cfg="${SCRIPT_DIR}/configs/${corpus}/${arm}.yaml"
  [[ -f "${cfg}" ]] || { echo "config not found: ${cfg}" >&2; exit 2; }
  dev="${GPU[$run]}"
  sess="msc_${run}"; log="/tmp/msc_${run}.log"
  if tmux has-session -t "${sess}" 2>/dev/null; then
    echo "SKIP ${sess}: tmux session already exists (kill it first to relaunch)" >&2
    continue
  fi
  tmux new-session -d -s "${sess}" \
    "cd ${REPO_ROOT} && LD_LIBRARY_PATH=${ENV_LIB} ${PY} experiments/run_single_experiment.py --config ${cfg} --device ${dev} > ${log} 2>&1"
  echo "launched ${sess}: corpus=${corpus} arm=${arm} gpu=${dev} log=${log}"
done
tmux ls | grep '^msc_' || true

#!/usr/bin/env bash
# mix_slp_ablation — eval-time 2x2 ablation of the emergent static-LP finding.
#
# Question: is MIX's above-chance 0-shot static link prediction (FINDINGS of
# multitask_ssl_rotation: mean AUC 0.759 vs <=0.467 for NM/CL/FP) genuinely
# TOPOLOGICAL (adjacency decoded from message passing) or a FEATURE artifact
# (feature homophily of the node bag)?
#
# Design: frozen 30k checkpoints (MIX treatment + NM control), static-LP ONLY,
# 0-shot, 4 eval datasets, under 4 eval-graph conditions:
#   none    -> unmodified graph                          (sanity anchor)
#   rewire  -> --ablate-edges rewire        (_ablE)  message passing sees a
#              random same-size edge set over the subgraph's node support;
#              the LP task still scores the TRUE held-out edges
#   permute -> --ablate-features permute    (_ablP)  feature rows shuffled
#              across nodes per subgraph; true edges kept
#   both    -> rewire + permute             (_ablPE)
#
# Interpretation matrix (see README.md): feature-homophily signal survives
# rewire / dies under permute; true-topology signal dies under rewire /
# survives permute. MIX surviving rewire => the emergent-sLP finding is a
# feature artifact (the gating verdict for the parallel training workstream).
#
# No retraining: ablations are per-subgraph eval-time augs (data/augment.py
# AblateEdges/AblateAllFeatures) composed by --ablate_* in experiments/params.py.
# All randomness is seeded: the runner passes --seed ${SEED} (default 0) which
# run_single_experiment.py feeds to torch/np/random; eval episodes are seeded
# by split name (sum(ord(split))), so episode content is IDENTICAL across
# conditions and any delta is attributable to the ablation.
#
# Usage (Tucker, from the exp/mix-slp-ablation worktree, prodigy env, tmux):
#   MODEL_LIST=scripts/experiments/setup/mix_slp_ablation/model_list.txt \
#     bash scripts/experiments/setup/mix_slp_ablation/run_2x2_slp.sh --gpus 0
#   CONDITIONS="none rewire" ... to run a subset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # setup/<name> is 4 deep
ML="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${ML}" ]] || { echo "model list not found: ${ML} (run make_model_list.sh)" >&2; exit 2; }

if [[ -z "${SKIP_CONDA:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
cd "${REPO_ROOT}"

RUNNER=scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
SEED="${SEED:-0}"
DATASETS="${DATASETS:-midterm,ukr_rus_twitter,covid19_twitter,twibot20}"
COMMON=(--model-list "${ML}" --python python3
        --data-root /dataMeR1/phil/data
        --datasets "${DATASETS}"
        --seed "${SEED}"
        --tasks slp --shots 0 --slp-n-query 4
        --continue-on-error)

run_condition() {  # $1 = label; $2.. = ablation flags
  local label="$1"; shift
  echo "=== mix_slp_ablation condition: ${label} (${*:-intact}) ==="
  python3 "${RUNNER}" "${COMMON[@]}" "$@" "${GPU_ARGS[@]}"
}

GPU_ARGS=("$@")  # pass --gpus ... straight through
CONDITIONS="${CONDITIONS:-none rewire permute both}"
for cond in ${CONDITIONS}; do
  case "${cond}" in
    none)    run_condition none ;;
    rewire)  run_condition rewire  --ablate-edges rewire ;;
    permute) run_condition permute --ablate-features permute ;;
    both)    run_condition both    --ablate-features permute --ablate-edges rewire ;;
    *) echo "unknown condition: ${cond}" >&2; exit 2 ;;
  esac
done

# collect all conditions into the analysis CSV (reads metrics_test.json directly;
# the shared parser's SLP_RE does not match the _ablE/_ablP/_ablPE tags)
python3 "${SCRIPT_DIR}/parse_slp_2x2.py" \
  --log-root "${REPO_ROOT}/log" \
  --out scripts/experiments/analysis/mix_slp_ablation/data/slp_ablation_2x2.csv

echo "MIX_SLP_ABLATION_2X2_DONE"

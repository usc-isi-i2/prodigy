#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
TARGET="${TARGET:?set TARGET}"
SHOT="${SHOT:?set SHOT}"
LABEL_SEED="${LABEL_SEED:?set LABEL_SEED}"
ARM="${ARM:?set ARM to pretrained or no_pretrain}"
GPU="${GPU:?set GPU to 2 or 3}"
RUN_STAMP="${RUN_STAMP:-20260830}"
STATE_ROOT="${STATE_ROOT:-$REPO_ROOT/state/rq1_native_icl}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/log/rq1_native_icl}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-/dataMeR1/phil/gfm/prodigy-rq1/state/rq1_label_efficiency_loto/pretrain}"

[[ "$TARGET" =~ ^(covid_political|election2020|ukr_rus_suspended|twibot20)$ ]]
[[ "$SHOT" =~ ^(1|3|5|10)$ ]]
[[ "$LABEL_SEED" =~ ^[0-4]$ ]]
[[ "$ARM" =~ ^(pretrained|no_pretrain)$ ]]
[[ "$GPU" =~ ^[23]$ ]]

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

run_id="rq1_icl_${TARGET}_${SHOT}shot_ls${LABEL_SEED}_${ARM}"
result="$STATE_ROOT/results/$TARGET/${SHOT}shot/label_seed_${LABEL_SEED}/$ARM/result.json"
log_data="$LOG_ROOT/eval/${run_id}_${RUN_STAMP}/data"
mkdir -p "$(dirname "$result")" "$STATE_ROOT/eval" "$LOG_ROOT/eval"
[[ -s "$result" ]] && { echo "SKIP completed $run_id"; exit 0; }

pretrained_args=()
checkpoint=""
if [[ "$ARM" == pretrained ]]; then
  checkpoint="$PRETRAIN_ROOT/rq1_loto_${TARGET}_pretrain_s0_20260828/state_dict"
  [[ -s "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 10; }
  pretrained_args=(--pretrained_model_run "$checkpoint")
fi

"$CONDA_PREFIX/bin/python" -u experiments/run_single_experiment.py \
  --config "$SCRIPT_DIR/base.yaml" \
  --dataset "$TARGET" --root "/dataMeR1/phil/data/$TARGET/graphs" \
  --graph_filename retweet_graph.pt --n_shots "$SHOT" --n_query 3 \
  --classification_support_from_train True --classification_support_cap "$SHOT" \
  --classification_support_seed "$LABEL_SEED" \
  --test_len_cap 125 --device "$GPU" --workers 8 --seed 0 \
  --prefix "$run_id" --timestamp "$RUN_STAMP" \
  --state_dir "$STATE_ROOT/eval" --log_dir "$LOG_ROOT/eval" \
  --tags rq1 native-icl zero-update "$TARGET" "${SHOT}shot" "label-seed-$LABEL_SEED" "$ARM" \
  "${pretrained_args[@]}"

metrics="$log_data/metrics_test_step0.json"
[[ -s "$metrics" ]] || { echo "missing metrics: $metrics" >&2; exit 11; }
"$CONDA_PREFIX/bin/python" - "$result" "$metrics" "$TARGET" "$SHOT" "$LABEL_SEED" "$ARM" "$checkpoint" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, metrics = Path(sys.argv[1]), Path(sys.argv[2])
target, shot, label_seed, arm = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
checkpoint = Path(sys.argv[7]) if sys.argv[7] else None
payload = {
    "protocol_version": "native-icl-train-support-500episodes-v1",
    "target": target, "shots_per_class": shot, "label_seed": label_seed,
    "model_seed": 0, "arm": arm, "downstream_updates": 0,
    "support_split": "train", "query_split": "test",
    "test_episodes": 500, "queries_per_class": 3,
    "checkpoint": str(checkpoint) if checkpoint else None,
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
    "test": json.loads(metrics.read_text()),
}
tmp = out.with_suffix(".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(out)
print(f"WROTE {out}")
PY

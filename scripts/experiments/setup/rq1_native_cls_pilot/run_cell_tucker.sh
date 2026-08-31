#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
TARGET="${TARGET:?set TARGET}"
BUDGET="${BUDGET:?set BUDGET}"
ARM="${ARM:?set ARM to pretrained or scratch}"
GPU="${GPU:?set GPU to 2 or 3}"
RUN_STAMP="${RUN_STAMP:-20260830}"
STATE_ROOT="${STATE_ROOT:-$REPO_ROOT/state/rq1_native_cls_pilot}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/log/rq1_native_cls_pilot}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-/dataMeR1/phil/gfm/prodigy-rq1/state/rq1_label_efficiency_loto/pretrain}"
SMOKE="${SMOKE:-0}"

[[ "$TARGET" =~ ^(covid_political|election2020|ukr_rus_suspended|twibot20)$ ]]
[[ "$BUDGET" =~ ^(10|100|1000)$ ]]
[[ "$ARM" =~ ^(pretrained|scratch)$ ]]
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

graph_root="/dataMeR1/phil/data/$TARGET/graphs"
run_id="rq1_native_${TARGET}_b${BUDGET}_${ARM}_s0"
train_name="${run_id}_${RUN_STAMP}"
train_state="$STATE_ROOT/train/$train_name"
result="$STATE_ROOT/results/$run_id/result.json"
mkdir -p "$STATE_ROOT/train" "$STATE_ROOT/eval" "$STATE_ROOT/results/$run_id" "$LOG_ROOT/train" "$LOG_ROOT/eval"

if [[ -s "$result" ]]; then
  echo "SKIP completed $run_id"
  exit 0
fi

pretrained_args=()
if [[ "$ARM" == pretrained ]]; then
  checkpoint="$PRETRAIN_ROOT/rq1_loto_${TARGET}_pretrain_s0_20260828/state_dict"
  [[ -s "$checkpoint" ]] || { echo "missing pretrained checkpoint: $checkpoint" >&2; exit 10; }
  pretrained_args=(--pretrained_model_run "$checkpoint")
fi

train_steps=1000
val_cap=32
test_cap=32
if [[ "$SMOKE" == 1 ]]; then
  train_steps=3
  val_cap=1
  test_cap=1
fi

if [[ ! -s "$train_state/state_dict" ]]; then
  "$CONDA_PREFIX/bin/python" -u experiments/run_single_experiment.py \
    --config "$SCRIPT_DIR/base.yaml" \
    --dataset "$TARGET" --root "$graph_root" --graph_filename retweet_graph.pt \
    --train_cap "$BUDGET" --dataset_len_cap "$train_steps" \
    --val_len_cap "$val_cap" --test_len_cap "$test_cap" \
    --eval_step "$([[ "$SMOKE" == 1 ]] && echo 1 || echo 100)" \
    --checkpoint_step "$([[ "$SMOKE" == 1 ]] && echo 1 || echo 100)" \
    --early_stopping_patience 3 --device "$GPU" --workers "$([[ "$SMOKE" == 1 ]] && echo 0 || echo 8)" \
    --prefix "$run_id" --timestamp "$RUN_STAMP" \
    --state_dir "$STATE_ROOT/train" --log_dir "$LOG_ROOT/train" \
    --tags rq1 native-cls label-efficiency seed0 "$TARGET" "budget-$BUDGET" "$ARM" \
    "${pretrained_args[@]}"
fi
[[ -s "$train_state/state_dict" ]] || { echo "missing selected checkpoint: $train_state/state_dict" >&2; exit 11; }

eval_id="${run_id}_test"
eval_log="$LOG_ROOT/eval/${eval_id}_${RUN_STAMP}/data"
if [[ ! -s "$eval_log/metrics_test_step0.json" ]]; then
  "$CONDA_PREFIX/bin/python" -u experiments/run_single_experiment.py \
    --config "$SCRIPT_DIR/base.yaml" \
    --dataset "$TARGET" --root "$graph_root" --graph_filename retweet_graph.pt \
    --train_cap "$BUDGET" --dataset_len_cap 1 --val_len_cap "$val_cap" --test_len_cap "$test_cap" \
    --eval_only True --eval_only_split test --eval_test_before_train True \
    --device "$GPU" --workers "$([[ "$SMOKE" == 1 ]] && echo 0 || echo 8)" \
    --pretrained_model_run "$train_state/state_dict" \
    --prefix "$eval_id" --timestamp "$RUN_STAMP" \
    --state_dir "$STATE_ROOT/eval" --log_dir "$LOG_ROOT/eval" \
    --tags rq1 native-cls label-efficiency seed0 test-once "$TARGET" "budget-$BUDGET" "$ARM"
fi
[[ -s "$eval_log/metrics_test_step0.json" ]] || { echo "missing test metrics: $eval_log/metrics_test_step0.json" >&2; exit 12; }

train_log_data="$LOG_ROOT/train/$train_name/data"
"$CONDA_PREFIX/bin/python" - "$result" "$eval_log/metrics_test_step0.json" "$train_state" "$train_log_data" "$TARGET" "$BUDGET" "$ARM" <<'PY'
import hashlib, json, re, sys
from pathlib import Path

out = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
train_state = Path(sys.argv[3])
train_log_data = Path(sys.argv[4])
target, budget, arm = sys.argv[5], int(sys.argv[6]), sys.argv[7]

val_rows = []
for path in train_log_data.glob("metrics_val_step*.json"):
    match = re.search(r"step(\d+)", path.name)
    row = json.loads(path.read_text())
    if match and "val_roc_auc" in row:
        val_rows.append((float(row["val_roc_auc"]), int(match.group(1))))
if not val_rows:
    raise RuntimeError(f"no validation ROC-AUC metrics under {train_log_data}")
selected_val, selected_step = max(val_rows)
checkpoint = train_state / "state_dict"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
payload = {
    "protocol_version": "native-cls-rocauc-val128-delta002-v2",
    "model_seed": 0,
    "label_seed": 0,
    "target": target,
    "budget_per_class": budget,
    "arm": arm,
    "native_classifier": True,
    "linear_probe": False,
    "n_way": 2,
    "n_shots": 5,
    "n_query": 5,
    "selection_metric": "val_roc_auc",
    "selected_val_roc_auc": selected_val,
    "selected_step": selected_step,
    "selected_checkpoint": str(checkpoint),
    "selected_checkpoint_sha256": digest,
    "test": json.loads(metrics_path.read_text()),
}
tmp = out.with_suffix(".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(out)
print(f"WROTE {out}")
PY

#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /dataMeR1/phil/gfm/prodigy-archnative
STATE_ROOT="$PWD/state/icl_arch_native_source_900_seed0" \
LOG_ROOT="$PWD/log/icl_arch_native_source_900_seed0" \
STEPS=900 CHECKPOINTS="20,60,100,300,900" GPUS_TEXT="2 3" \
bash scripts/experiments/setup/icl_arch_matrix/run_native_source_training_tucker.sh

STATE_ROOT="$PWD/state/icl_arch_native_source_900_seed0" \
OUT_ROOT="$PWD/log/icl_arch_native_source_900_seed0_eval" \
STEPS_TEXT="20 60 100 300 900" GPUS_TEXT="2 3" \
bash scripts/experiments/setup/icl_arch_matrix/run_native_source_cls_evaluation_tucker.sh

python - <<'PY'
from pathlib import Path

root = Path("log/icl_arch_native_source_900_seed0_eval/results")
files = list(root.glob("*.jsonl"))
rows = sum(sum(1 for line in path.open(encoding="utf-8") if line.strip()) for path in files)
if len(files) != 52 or rows != 260:
    raise SystemExit(f"incomplete native CLS evaluation: files={len(files)}/52 rows={rows}/260")
Path("log/icl_arch_native_source_900_seed0_eval/COMPLETE").write_text(
    f"files={len(files)}\nrows={rows}\n", encoding="utf-8"
)
print(f"native source sweep complete: files={len(files)} rows={rows}")
PY

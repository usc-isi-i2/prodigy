#!/usr/bin/env bash
# Wait for both GPU paper queues to pass their exact audits, then use GPUs 0-3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGSHIP_ROOT="${FLAGSHIP_ROOT:-/dataMeR1/phil/gfm/prodigy-paper-remainder-opt/log/paper_flagship_ladders/20260908}"
CORE_ROOT="${CORE_ROOT:-/dataMeR1/phil/gfm/prodigy-paper-remainder-opt/log/paper_three_seed_remainder/20260908/core_nm_evaluation}"
GPUS_TEXT="${GPUS:-0 1 2 3}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

while true; do
  if "${CONDA_PREFIX}/bin/python" - "$FLAGSHIP_ROOT" "$CORE_ROOT" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys

flagship, core = map(Path, sys.argv[1:])
training = json.load(open(flagship / "seeds_1-2/status.json", encoding="utf-8"))
if training.get("status") != "complete" or len(training.get("finished", [])) != 96:
    raise SystemExit(1)
if any(row.get("exitcode") != 0 for row in training["finished"]):
    raise SystemExit(1)
nm = json.load(open(flagship / "nm_evaluation/status.json", encoding="utf-8"))
if nm.get("status") != "complete":
    raise SystemExit(1)
nm_rows = []
for path in sorted((flagship / "nm_evaluation/cells").glob("*/*.json")):
    nm_rows.append(json.load(open(path, encoding="utf-8")))
nm_keys = [(row.get("model_id"), row.get("target")) for row in nm_rows]
if len(nm_rows) != 864 or len(set(nm_keys)) != 864:
    raise SystemExit(1)
if any(row.get("protocol") != "nmi_fixed_nm_v1" or int(row.get("episodes", 0)) != 512
       or not all(math.isfinite(float(row[key])) for key in ("roc_auc", "accuracy", "loss"))
       for row in nm_rows):
    raise SystemExit(1)
for target in {row["target"] for row in nm_rows}:
    if len({row["fingerprint"] for row in nm_rows if row["target"] == target}) != 1:
        raise SystemExit(1)
cls = flagship / "classification_evaluation/classification_long.tsv"
with open(cls, encoding="utf-8") as handle:
    if sum(1 for _ in csv.DictReader(handle, delimiter="\t")) != 600:
        raise SystemExit(1)
core_status = json.load(open(core / "status.json", encoding="utf-8"))
if core_status.get("status") != "complete" or core_status.get("audit", {}).get("cells") != 738:
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 60
done

stable=0
while (( stable < 4 )); do
  gpu_csv="$(tr ' ' ',' <<< "$GPUS_TEXT")"
  if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu_csv" |
      awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  sleep 30
done

GPUS="$GPUS_TEXT" exec bash "$SCRIPT_DIR/run_tucker.sh"

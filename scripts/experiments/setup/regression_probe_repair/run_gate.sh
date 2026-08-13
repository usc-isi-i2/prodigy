#!/usr/bin/env bash
# THE GATE — reproduce a number we already have, before trusting anything new.
#
# `analysis/evaluation/shared_task_tables/node_regression/data/features_only_floor.csv` records the raw-feature
# floor at 10 shots. This runs OUR probe implementation on the SAME raw features and
# must land on those values. If it does not, the protocol differs from the published
# floor and every encoder-vs-floor comparison built on it would be invalid.
#
# Reference values (features_only_floor.csv, midterm):
#   followers_count   0.2597
#   statuses_count    0.0546
#   account_age_days  0.0398
#
# midterm is used because it is the smallest graph carrying the full profile panel,
# so the gate costs a couple of minutes rather than an hour. No checkpoint, no GPU:
# --no-encoder scores raw features only.
#
#   bash run_gate.sh                 # run and print the comparison
#   TOL=0.02 bash run_gate.sh        # loosen the tolerance
#
# Exit 0 = PASS, 1 = mismatch (STOP), 2 = could not run.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/gate_out}"
TOL="${TOL:-0.02}"

python3 scripts/eval/regression_probe_sweep.py \
  --graph "${DATA_ROOT}/midterm/graphs/retweet_graph_parquet.pt" \
  --dataset midterm \
  --no-encoder \
  --out-dir "${OUT_DIR}" \
  --targets followers_count,statuses_count,account_age_days \
  --transform log1p --shots 10 --n-query 12 --episodes 500 --alpha 1.0 \
  --device cpu || { echo "GATE: sweep failed" >&2; exit 2; }

python3 - "${OUT_DIR}/midterm__reg_probe.csv" "${TOL}" <<'PYCHECK'
import csv, sys
from pathlib import Path

csv_path, tol = Path(sys.argv[1]), float(sys.argv[2])
ref_path = Path("scripts/experiments/analysis/evaluation/shared_task_tables/node_regression/data/features_only_floor.csv")
ref = {}
for r in csv.DictReader(ref_path.open()):
    if r["dataset"] == "midterm" and r["shots"] == "10":
        ref[r["target"]] = float(r["spearman"])

rows = [r for r in csv.DictReader(csv_path.open()) if r["model"] == "__features_only__"]
if not rows:
    print("GATE: no floor rows emitted", file=sys.stderr); raise SystemExit(2)

print(f"\n{'target':<20}{'published':>11}{'ours':>11}{'delta':>10}")
bad = 0
for r in rows:
    t = r["target"]
    if t not in ref:
        print(f"{t:<20}{'--':>11}{float(r['spearman']):>11.4f}{'  (no ref)':>10}")
        continue
    d = float(r["spearman"]) - ref[t]
    flag = "" if abs(d) <= tol else "   <-- MISMATCH"
    bad += abs(d) > tol
    print(f"{t:<20}{ref[t]:>11.4f}{float(r['spearman']):>11.4f}{d:>+10.4f}{flag}")

print(f"\nGATE: {'PASS' if not bad else 'FAIL'} (tolerance {tol})")
raise SystemExit(1 if bad else 0)
PYCHECK

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/log/prodigy_source_lattice_lp}"
MODEL_LIST="${OUT_ROOT}/model_list.tsv"
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/worker_logs"

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline

python "${ROOT}/scripts/experiments/setup/prodigy_source_lattice_lp/build_model_list.py" --output "${MODEL_LIST}"

TARGETS=(
  "ukr_rus_twitter:/dataMeR1/phil/data/ukr_rus_twitter/graphs/retweet_graph_parquet.pt"
  "covid19_twitter:/dataMeR1/phil/data/covid19_twitter/graphs/retweet_graph_parquet.pt"
  "midterm:/dataMeR1/phil/data/midterm/graphs/retweet_graph_parquet.pt"
  "covid_political:/dataMeR1/phil/data/covid_political/graphs/retweet_graph.pt"
  "election2020:/dataMeR1/phil/data/election2020/graphs/retweet_graph.pt"
  "ukr_rus_suspended:/dataMeR1/phil/data/ukr_rus_suspended/graphs/retweet_graph.pt"
  "twibot20:/dataMeR1/phil/data/twibot20/graphs/retweet_graph.pt"
  "cp_hk_twitter:/dataMeR1/phil/data/cp_hk_twitter/graphs/retweet_graph.pt"
  "facebook_page_reference:/dataMeR1/phil/data/facebook_page_reference/graphs/page_reference_graph.pt"
)

pids=()
for worker in 0 1 2 3; do
  (
    for index in "${!TARGETS[@]}"; do
      (( index % 4 == worker )) || continue
      target="${TARGETS[$index]%%:*}"
      graph="${TARGETS[$index]#*:}"
      python "${ROOT}/scripts/eval/pair_link_sweep.py" \
        --graph "${graph}" --dataset "${target}" --model-list "${MODEL_LIST}" \
        --edge-split "/dataMeR1/phil/gfm/mixture-scaling-graphmae/state/social9_source_lattice/_cache/${target}_edge_split_s0.pt" \
        --out-dir "${OUT_ROOT}/data" --negative-kinds degree_matched \
        --max-positives 2000 --n-hop 1 --batch-size 256 --seed 0 \
        --device "cuda:${worker}" --resume
    done
  ) >"${OUT_ROOT}/worker_logs/gpu_${worker}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
python - "${OUT_ROOT}/data" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted(root.glob("*__pair_lp.csv"))
counts = {}
for path in files:
    rows = list(csv.DictReader(path.open()))
    valid = [r for r in rows if r["model"] != "__floor__" and r["negative_kind"] == "degree_matched"]
    counts[path.name] = len({r["model"] for r in valid})
print(counts)
if len(files) != 9 or any(value != 54 for value in counts.values()):
    raise SystemExit("incomplete 54 x 9 lattice")
PY
exit "${failed}"

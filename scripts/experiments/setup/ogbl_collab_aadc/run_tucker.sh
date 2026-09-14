#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

branch="$(git branch --show-current)"
[[ "$branch" == "codex/ogbl-collab-aadc" ]] || { echo "unexpected branch $branch" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "refusing dirty worktree" >&2; exit 1; }

upstream_revision="b499c2046cfe76448545dfe08fad9effb58dd076"
run_root="/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204"
[[ ! -e "$run_root" ]] || { echo "refusing to overwrite $run_root" >&2; exit 1; }
mkdir -p "$run_root"
git clone https://github.com/anmo-fish/aa-dc.git "$run_root/upstream"
git -C "$run_root/upstream" checkout --detach "$upstream_revision"
actual="$(git -C "$run_root/upstream" rev-parse HEAD)"
[[ "$actual" == "$upstream_revision" ]] || { echo "upstream revision mismatch" >&2; exit 1; }
ln -s /dataMeR1/phil/data/ogb "$run_root/dataset"

revision="$(git rev-parse HEAD)"
start="$(date +%s)"
cd "$run_root"
python upstream/aa_dc.py --use_gate 1 --gate_mode progressive --auto_gate 1 \
  --use_l3 1 --rescue_mode anchor --auto_beta 1 --name ogbl-collab \
  --debug 0 --progress 0 2>&1 | tee raw.log
end="$(date +%s)"
cd /dataMeR1/phil/gfm/prodigy-ogbl-collab-aadc
python scripts/experiments/setup/ogbl_collab_aadc/track_result.py \
  --log "$run_root/raw.log" --out "$run_root" --revision "$revision" \
  --elapsed-seconds "$((end - start))" --wandb-mode online

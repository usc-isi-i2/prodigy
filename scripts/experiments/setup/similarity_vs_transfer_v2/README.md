# Similarity vs. transfer v2: extended predictor extraction

This is the second experiment in the `similarity_vs_transfer` line.  The first
experiment asked whether six graph distances tracked a small transfer pilot;
v2 evaluates predictors against the complete directed 9×9 single-source NM
matrix and adds predictors closer to what a message-passing GNN observes.

The laptop analysis consumes the committed transfer matrix and graph-divergence
artifact. The Tucker job generates user overlap, feature skew, and distances in
raw-center, sampled-neighbor-mean, concatenated, local-structure, and
embedding-topic spaces. The complete run was generated on 2026-08-07.

## Worktree and branch

- branch: `experiment/similarity-vs-transfer-v2`
- laptop worktree: `/Users/philipp/projects/gfm/prodigy-simtransfer-v2`
- recommended Tucker worktree: `/dataMeR1/phil/gfm/prodigy-simtransfer-v2`

Do not run the Tucker extraction from a checkout that is serving another job.

## Reproduce the current 9×9 leaderboard locally

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/similarity_vs_transfer_v2/analyze_predictors.py
```

## Proper final-core AUC extension

The strict 243-cell metric-preserving evaluation completed on 2026-08-08 in a
separate worktree so it could not change code under any other evaluation:

- branch: `experiment/final-core-auc-grid`
- commit: `62d9e31`
- Tucker worktree: `/dataMeR1/phil/gfm/prodigy-final-core-auc`
- source summary: `/dataMeR1/phil/gfm/prodigy-final-core-auc/log/final_core_auc/production/bs32/summary`
- completion marker: `2026-08-08T16:01:16Z`

The successful launch used `MAX_EXISTING_GPU_MIB=8192` and
`MIN_HOST_RESERVE_GIB=128`. These are resource gates only; they do not change
episodes, checkpoints, predictions, or metrics. The evaluator first passed a
10-cell smoke, then completed all 243 specialist cells with the fixed episode
fingerprints from the published final-core ledger.

The raw aggregate and its provenance are committed under
`analysis/similarity_vs_transfer_v2/data/final_core_auc/raw/`. Rebuild all local
evidence with Homebrew Python 3.11:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/similarity_vs_transfer_v2/import_final_core_auc.py

/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/similarity_vs_transfer_v2/analyze_final_core_matrix.py \
  --cells scripts/experiments/analysis/similarity_vs_transfer_v2/data/final_core_auc/specialist_cells_three_seed.csv \
  --metric roc_auc_ovr_macro \
  --out-dir scripts/experiments/analysis/similarity_vs_transfer_v2/data/final_core_auc/predictors \
  --permutations 9999 --seed 20260808

/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/similarity_vs_transfer_v2/compare_final_core_historical.py \
  --final-core scripts/experiments/analysis/similarity_vs_transfer_v2/data/final_core_auc/specialist_cells_three_seed.csv \
  --final-core-metric roc_auc_ovr_macro \
  --out-dir scripts/experiments/analysis/similarity_vs_transfer_v2/data/final_core_auc/comparison

/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/similarity_vs_transfer_v2/fit_candidate_models.py \
  --transfer scripts/experiments/analysis/similarity_vs_transfer_v2/data/final_core_auc/transfer_matrix_three_seed_mean_long.csv \
  --metric roc_auc_ovr_macro \
  --out-dir scripts/experiments/analysis/similarity_vs_transfer_v2/data/final_core_auc/models

/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/similarity_vs_transfer_v2/analyze_auc_predictability.py \
  --transfer scripts/experiments/analysis/similarity_vs_transfer_v2/data/final_core_auc/transfer_matrix_three_seed_mean_long.csv \
  --metric roc_auc_ovr_macro \
  --out-dir scripts/experiments/analysis/similarity_vs_transfer_v2/data/final_core_auc/predictability
```

## Generate the missing feature/user predictors on Tucker

After the branch is committed and pushed, create a dedicated Tucker worktree:

```bash
cd /dataMeR1/phil/gfm/prodigy
git fetch origin
git worktree add ../prodigy-simtransfer-v2 experiment/similarity-vs-transfer-v2
cd ../prodigy-simtransfer-v2
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python scripts/experiments/setup/similarity_vs_transfer_v2/compute_extended_predictors_tucker.py \
  --data-root /dataMeR1/phil/data \
  --out scripts/experiments/analysis/similarity_vs_transfer_v2/data/extended_predictors.json
```

The extractor writes only a final JSON artifact. It loads graphs one at a time,
samples 2,000 feature-bearing centers, uses undirected one-hop fanout 100 (the
historical PRODIGY convention), and hashes normalized user IDs before overlap
calculation. Facebook↔Twitter user overlap is reported unavailable rather than
as zero. Temporal distances are intentionally not guessed: final graph objects
do not expose comparable event timestamps for all nine corpora, so a separate
raw-source temporal extraction is still required.

# Similarity vs. transfer v2: extended predictor extraction

This is the second experiment in the `similarity_vs_transfer` line.  The first
experiment asked whether six graph distances tracked a small transfer pilot;
v2 evaluates predictors against the complete directed 9×9 single-source NM
matrix and adds predictors closer to what a message-passing GNN observes.

The laptop analysis consumes the already committed transfer matrix and graph
divergence artifact.  The optional Tucker job generates the missing expensive
predictors: exact user overlap, feature skew summaries, and distributional
distances in raw-center, sampled-neighbor-mean, and concatenated spaces.

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

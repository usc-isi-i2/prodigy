# Canonical NM source-by-stage replay

Compare the frozen seed-0 Ukraine/Hong Kong checkpoints on every original
canonical test input: two targets, 512 episodes and 61,440 queries each.
No training, new episode sampling, support replacement or head fitting.

The identical fixed head scores each query by mean cosine to each class's three
individually normalized pre-metagraph support embeddings. A normalized mean
prototype is a secondary sensitivity head. Final predictions use the original
metagraph, learned episode-position label vectors and decoder. Raw neighborhood
mean geometry comes from the existing full-input analysis; missing raw summaries
must remain excluded from strict all-candidate comparisons, never imputed to zero.
Head failures do not prove absence of information in an embedding. Differences
between checkpoints or stages are descriptive, not causal mediation.

The replay reads existing compact inputs and indexes required feature rows from
a read-only memory map of the exact merged feature artifact. It never loads the
whole graph or builds adjacency. All 163 available HK-model episode embeddings
are reused. New representations and logits are saved per original batch.

Validity gates, fixed before outcomes:

- Verify all compact-file and reused-cache SHA-256 hashes.
- Match both complete realized-input hashes to the original canonical receipt.
- Strict checkpoint loading and unchanged model-state hashes.
- All original predicted anchor IDs and correctness outcomes must match exactly;
  true-class probability error must be below 1e-5.
- A full-batch witness for each model/target checks all 32 episodes, with absolute
  pre-metagraph tolerance 1e-5 and logit tolerance 1e-4.
- Preserve labels at their original within-batch positions.

Runtime:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python scripts/experiments/setup/nm_source_stage_audit/run.py --self-test
python scripts/experiments/setup/nm_source_stage_audit/run.py --dry-run --out /dataMeR1/phil/gfm/error_audit/nm_source_stage_20260908
python -u scripts/experiments/setup/nm_source_stage_audit/run.py --device cuda:0 --out /dataMeR1/phil/gfm/error_audit/nm_source_stage_20260908
```

Run from its isolated worktree in tmux; only owned available GPUs may be used.
All paths, CPU threads and device are overrideable. The output directory must be
new. Private per-query rows and embeddings stay outside git. Findings belong in
`scripts/experiments/analysis/evaluation/error_audit/`.

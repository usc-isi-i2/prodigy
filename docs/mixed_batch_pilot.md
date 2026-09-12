# Balanced mixed-batch pilot

This is distinct from alternating source updates. Each optimizer step computes BCE on 512 positive edges from each graph (plus five source-confined uniform exact-nonedge negatives per positive) at identical model weights, averages the two losses, backpropagates once, clips the combined gradient, and takes one shared AdamW step. Positives wrap epoch boundaries to keep exactly 512 per source. The alternating reference uses 1,024 positives from one source per step, with partial final epoch batches; exposure is approximately matched per update, with the small boundary difference recorded here.

Three seed-0 diagnostic pairs: Ukraine/Russia + Facebook pages; COVID-19 + Midterm; COVID political + Ukraine suspended. Fresh 1536→256→256 MLP and learned scalar decoder bias. Other hyperparameters match interleaved training. Source sub-batch counts in `updates_per_source` now count participation: both graphs participate every optimizer step; each contributes 512 positives. Checkpoint step remains the number of optimizer updates, not source participations.

Validation every 2,000 updates on both sources; select minimum mean BCE, stop on three stale checks (min delta 1e-4; stale counting starts after 2,500 updates). 100,000-total-update cap. Equal stopping rules do not guarantee equal final compute: compare selected step counts and convergence flags. Eight evaluation targets, Election excluded, yield 24 evaluation cells; six unseen targets per pair. Same cached graph context and evaluation pairs as before. No cross-graph negatives or new graph artifacts.

Run from a dedicated Tucker worktree in tmux:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
bash scripts/run_mixed_batch_pilot.sh /dataMeR1/phil/gfm/mixture-scaling/state/mixed_batch_pilot_s0
```

Plan phase: `python -m mixture_scaling.mixed_batch_pilot plan --root <root>`. Existing runs are never replaced. A shared queue on GPUs 0–3 trains three models, then evaluates. The generic aggregate receipt's `interleaved_models` field records the number of models; `run_protocol.schedule` identifies this variant as `balanced_mixed_batches`.

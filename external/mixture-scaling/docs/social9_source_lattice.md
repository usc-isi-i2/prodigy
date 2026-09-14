# Social-9 native GraphSAGE objective lattice

This experiment trains the same native one-layer 768→256 GraphSAGE encoder on
the nine social graphs used by the source-transfer matrix. It covers all 9
specialists, 36 unordered pairs, and 9 everything-minus-one mixtures at seed 0.

Two objective passes use the identical model lattice and stopping rule:

1. source-confined link prediction with five sampled negatives per positive;
2. GraphMAE masked-feature reconstruction with a 50% node mask and scaled
cosine error (alpha 2).

The LP pass uses AdamW at `5e-4`; the initial `1e-3` gate produced repeatable
large loss oscillations in the eight-source mixture. GraphMAE retains `1e-3`.
Both objectives clip the global gradient norm at 1.0 and log the unclipped norm.

Mixtures receive uniform round-robin source updates. Validation runs every 250
updates. Training cannot stop before 2,000 updates; it stops after eight checks
without a 0.25% relative improvement, or at 10,000 updates. The absolute best
validation checkpoint is tracked independently from the patience reference.

Every run retains `best.pt`, periodic and terminal checkpoints, optimizer state,
the full pretraining-model state, the downstream GraphSAGE encoder state,
effective metadata, JSONL curves, a summary, and an offline W&B run when W&B is
installed. Four long-lived workers load their required graphs once and reuse
them across assigned models. LP's deterministic edge partitions are cached
under the experiment state with a process lock, so expensive canonicalization
happens once per graph rather than once per GPU. GraphMAE bypasses that LP-only
work and loads the observed topology directly. Only Tucker GPUs 0–3 are accepted.

Before each full pass, run the three-model curve gate:

```bash
bash scripts/run_social9_lattice_tucker.sh lp gate
bash scripts/run_social9_lattice_tucker.sh lp full
bash scripts/run_social9_lattice_tucker.sh graphmae gate
bash scripts/run_social9_lattice_tucker.sh graphmae full
```

The gate covers a large specialist (`covid19_twitter`), a heterogeneous pair
(`ukr_rus_suspended` + `facebook_page_reference`), and the heterogeneous
eight-source leave-COVID-out mixture. Inspect training and validation curves,
per-source validation losses, throughput, GPU utilization, early-stop steps,
NaNs, and worker failures before releasing the corresponding full pass.

Evaluation covers 270 frozen linear-probe classification cells (54 models by
five labeled targets) and 324 repaired static-link cells (54 by six eligible
targets). Static LP uses 2,000 held-out positives, degree-matched negatives,
validation-locked score orientation, both-endpoint cosine scoring, and explicit
leakage and endpoint-sensitivity gates. `mixture_scaling.aggregate_lattice`
refuses to emit final TSVs unless every model, checkpoint, offline W&B history,
and evaluation cell is present and valid.

## Completion receipt (2026-09-06)

The seed-0 LP pass and then the seed-0 GraphMAE pass completed on Tucker GPUs
0–3. Training used revision `cb8d334`; the final evaluator and aggregation used
revision `67a4fa4`. The full state remains under
`/dataMeR1/phil/gfm/mixture-scaling-graphmae/state/social9_source_lattice`, with
logs and evaluation JSON under the sibling `log/` and `results/` trees.

- LP: 54/54 models, 2,500–6,000 updates (mean 4,023), 5–7 checkpoints/model.
- GraphMAE: 54/54 models, 2,250–6,000 updates (mean 3,565), 5–7 checkpoints/model.
- Every best and terminal checkpoint contains encoder weights, full pretraining
  weights, optimizer state, step, and effective metadata; every run has an
  offline W&B history.
- All 1,188 downstream cells completed: 540 classification and 648 repaired
  static-link evaluations across both objectives.
- Static-link evaluation reconstructs the exact deterministic 85/15 LP split
  cached during LP training. Across all 648 cells, holdout leakage was zero,
  endpoint sensitivity was at least 0.99925, and endpoint-permutation AUC had
  mean 0.4965 (range 0.4789–0.5184).
- All recorded training losses, validation losses, gradients, and downstream
  AUCs were finite. No run selected its first validation check as best; median
  start-to-best validation improvement was 1.45% for LP and 1.05% for GraphMAE.
  Worst final-to-best drift was 1.02% and 0.25%, respectively.

Versioned aggregate evidence is in
`results/social9_source_lattice/aggregated/`: one training TSV, classification
TSV, static-link TSV, and completion JSON per objective. Raw checkpoints and
offline W&B directories intentionally remain on Tucker.

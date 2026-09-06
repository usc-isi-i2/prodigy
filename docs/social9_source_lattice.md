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

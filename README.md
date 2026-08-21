# Mixture Scaling

Controlled experiments comparing PRODIGY with plain GraphSAGE as the scale and
diversity of a graph pretraining mixture change.

## Research questions

1. How does downstream adaptation efficiency change with pretraining-mixture
   breadth and diversity?
2. How do those relationships change with model scale?
3. Can the resulting scaling relationships select a compute-efficient mixture
   that beats naive choices?

## RQ1 comparison

The initial table has three model families:

| Family | Pretraining | Comparable downstream evaluation |
|---|---|---|
| `sage_scratch` | none | frozen/random probe and fine-tuning |
| `sage_rw_ssl` | random-walk positives with negative sampling | linear probe and fine-tuning |
| `prodigy_nm` | neighbor matching | linear probe and fine-tuning |

PRODIGY's native in-context evaluation is reported separately because it is not
equivalent to fitting a linear head or fine-tuning GraphSAGE.

All primary rows use the same GraphSAGE depth and width, source mixtures, held-out
targets, label subsets, and seed. Pretrained rows train to validation convergence;
checkpoint trajectories are retained for matched-compute comparisons.

## First pilot

`configs/twibot20_sage_ssl_pilot.yaml` specifies a plain GraphSAGE random-walk SSL
convergence run on TwiBot-20. It deliberately contains no PRODIGY pooling,
metagraph, label embeddings, or episodic classification machinery.

The pilot is centered on the existing 2.5k evidence. It saves checkpoints at 100,
300, 600, 900, 1.5k, 2.5k, 5k, and 10k updates.
Convergence is selected only from a fixed validation stream; test metrics are read
once after selection.

## Repository layout

```text
configs/       versioned experiment definitions
src/           model, SSL objective, training, and evaluation code
tests/         protocol and implementation checks
results/       derived tables and figures (not raw checkpoints)
```

Large graph artifacts, checkpoints, and logs stay on Tucker under `/dataMeR1`.

Verified aggregate tables and the current RQ1 interpretation live in
`results/RESULTS.md`. Rebuild all derived tables from the four aggregate inputs with:

```bash
PYTHONPATH=src python -m mixture_scaling.analyze --root .
PYTHONPATH=src python -m mixture_scaling.plot_results --root .
```

## Primary overnight study

Seven seed-0 single-source specialists and seven leave-one-target-out six-source
mixtures are trained with uniform source rotation. For each labeled target, the
target specialist and its target-held-out mixture are evaluated at steps 100, 300,
900, and 2,500 with the same ten labeled nodes per class. The full cross-source
matrix and additional seeds follow only after this primary gate is complete.

On Tucker, `scripts/pipeline_tucker.sh` waits for GPUs 2–3, builds missing citation
artifacts non-destructively, runs tests and a two-step smoke, then executes the primary
training/evaluation/aggregation sequence. Every stage stops the pipeline on failure.

## Follow-up sweeps

After the primary gate, `scripts/pipeline_followups_tucker.sh` runs three declared
extensions on GPUs 2–3: the complete 14-model by 7-target seed-0 transfer matrix;
the held-out mixture ladder at sizes 2–5; and full primary replications at seeds 1
and 2. The ladder order is the graph order in `configs/graphs.yaml` with the held-out
target removed. Its size-1 and size-6 endpoints reuse the primary specialist and
leave-one-out models, so only 18 distinct intermediate mixtures require new training.

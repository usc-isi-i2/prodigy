# Five-labeled-graph mixture diversity at 500 steps

Seed-0 screening experiment for the fixed-compute question: how does transfer to an
unseen classification graph change as the number of labeled pretraining graphs grows?

The five sources/targets are `covid_political`, `election2020`,
`facebook_page_reference`, `ukr_rus_suspended`, and `twibot20`. Training covers every
nonempty proper subset: 5 singletons, 10 pairs, 10 triples, and 5 four-source mixtures.
The 30 physical checkpoints are evaluated only on absent targets, yielding 75 held-out
CLS cells (15 per target). Every model gets exactly 500 optimizer steps; evaluation
uses 500 deterministic 10-shot episodes per target.

The protocol matches final-core (`static_train`, real static split, 2-hop `9,9`,
101-node cap, batch size 4, learning rate 0.002). Training uses two loader workers and
launches every model in a fresh Python process. This matches the process isolation of
earlier successful sweeps and prevents worker state from crossing model boundaries.
Each model also has a 30-minute timeout, so a loader failure cannot hang the sweep.

```bash
python3 scripts/experiments/setup/labeled_mixture_diversity_cls500/make_plan.py --check
DRY_RUN=1 GPUS="0 1" bash scripts/experiments/setup/labeled_mixture_diversity_cls500/run_train_tucker.sh
GPUS="0 1" bash scripts/experiments/setup/labeled_mixture_diversity_cls500/run_train_tucker.sh
```

`run_pipeline_tucker.sh` trains and then evaluates all 75 valid held-out cells. The
evaluator loads each target graph once per shard, checks that all models consume 500
paired episodes, fingerprints those episodes, and writes restart-safe JSONL results.

```bash
GPUS="0 1" bash scripts/experiments/setup/labeled_mixture_diversity_cls500/run_pipeline_tucker.sh
```

This first pass estimates diversity at fixed compute. Fixed per-source exposure and
long single-source compute controls remain follow-ups if the seed-0 curve is material.

Endpoint controls reuse the five singleton checkpoints as target-only references,
train one fixed-compute all-five checkpoint, and evaluate both endpoints on each target:

```bash
bash scripts/experiments/setup/labeled_mixture_diversity_cls500/run_controls_tucker.sh
```

## Post-500 convergence trajectory

`continue_train.py` restores the step-500 model and AdamW state for all 31 physical
models, then trains 500 additional steps with a common fresh episode stream (seed
500). Resetting the stream uniformly is intentional: multiprocessing prefetch makes
the exact next episode unknowable for 21 of the original two-worker runs. The fixed
500-step result remains the primary fixed-compute estimand; this continuation asks
whether held-out CLS changes with further optimization.

The continuation saves local steps 250 and 500, corresponding to global training
steps 750 and 1,000. `run_trajectory_eval_tucker.sh` evaluates both checkpoints on
all 75 held-out cells and all 10 endpoint-control cells.

On Tucker, the original checkpoints remain in the completed experiment worktree:

```bash
SOURCE_STATE_ROOT=/dataMeR1/phil/gfm/prodigy-mixdiv2k/state \
  bash scripts/experiments/setup/labeled_mixture_diversity_cls500/run_continue_tucker.sh
bash scripts/experiments/setup/labeled_mixture_diversity_cls500/run_trajectory_eval_tucker.sh
```

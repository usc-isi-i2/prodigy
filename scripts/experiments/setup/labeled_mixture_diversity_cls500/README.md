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

# Held-out classification mixture diversity at 2k steps

This experiment asks whether few-shot node-classification performance changes with
pretraining mixture diversity under a fixed 2,000-step compute budget.

## Estimand and scope

The scope is the four labeled targets contained in the existing disjoint all-eight
social-media merge:

- `covid_political`
- `election2020`
- `ukr_rus_suspended`
- `twibot20`

For each target, the target component is excluded and every unordered combination of
1, 2, 3, or 4 of the other seven graphs is used for neighbor-matching pretraining.
This gives 98 model-target comparisons per target and 392 comparisons total:

| Donors in mixture | Combinations per target | Target-model comparisons |
|---:|---:|---:|
| 1 | 7 | 28 |
| 2 | 21 | 84 |
| 3 | 35 | 140 |
| 4 | 35 | 140 |

Identical donor sets are target-independent, so their checkpoints are reused. There
are only `C(8,1)+...+C(8,4) = 162` physical training runs (324,000 optimizer steps),
not 392 redundant runs. All combinations are exhaustive, not sampled. Within each target and mixture size,
every eligible donor therefore occurs equally often. The primary comparison is the
mean held-out 10-shot classification F1 at each mixture size, first per target and then
with target fixed effects.

Every model receives exactly 2,000 optimizer steps. Sources are selected uniformly
within a mixture, so expected exposure is `2000 / k` episodes per donor. The result is
the effect of diversity **at fixed total compute**; it does not separate diversity from
reduced per-donor exposure. A fixed-exposure follow-up would need `2000 * k` steps.

The standalone labeled graphs outside the all-eight social-media artifact
(`facebook_page_reference`, `cora`, `pubmed`, and the synthetic probes) are deliberately
out of scope: they are valid downstream transfer targets, but they cannot themselves
be held out from this eight-source donor universe. They can be added later as external
targets, which answers a different question.

## Leakage and protocol controls

Training reads the all-eight artifact but restricts eligible `graph_id`s to the donor
set. Episodes are confined to one source component and the merge has no cross-source
edges, so the held-out target contributes neither centers nor neighborhood nodes.

All arms use the established fair two-hop NM protocol: fanouts `9,9`, 101-node cap,
one-hop positive walks, balanced source selection, 30-way/3-shot/4-query episodes,
seed 0, and the same `256 · S,U,M` GraphSAGE encoder. Evaluation is 10-shot CLS on the
held-out target only, using the repository's fixed split-derived episode stream.

## Files

- `make_plan.py` owns the exhaustive combination plan and validates balance/leakage.
- `manifest.tsv` is the 162-run physical training plan; `evaluation_manifest.tsv`
  maps those reusable checkpoints to all 392 valid held-out-target comparisons.
- `base_train.yaml` freezes the common 2k protocol.
- `run_sweep.py` reuses one in-memory copy of the approximately 111 GB all-eight graph
  across many models, avoiding a graph reload for every 2k run.
- `run_train_tucker.sh` optionally shards the plan across GPUs 0 and 1.
- `make_model_lists.py` resolves only complete `state_dict_2000.ckpt` files.
- `eval_cls_tucker.sh` evaluates each model only on its held-out target and merges the
  results into the shared classification table.

## Validate and dry-run locally

```bash
python3 scripts/experiments/setup/nm_mixture_diversity_heldout_cls_2k/make_plan.py --check
pytest -q scripts/experiments/setup/nm_mixture_diversity_heldout_cls_2k/tests
```

The launcher needs Tucker's conda installation, so dry-run it on Tucker:

```bash
DRY_RUN=1 GPUS="0" LIMIT=4 \
  bash scripts/experiments/setup/nm_mixture_diversity_heldout_cls_2k/run_train_tucker.sh
```

## Tucker execution

Use a dedicated worktree and check `tmux ls` before changing it. The default uses only
GPU 0 and loads the large merge once. Use `GPUS="0 1"` only after confirming host RAM
can hold two copies; GPUs other than 0 and 1 are currently off limits.

If an owned GPU is busy, `wait_and_train_tucker.sh` waits for it to remain below
1,000 MiB and 10% utilization for four consecutive 30-second polls before launching.
This avoids stealing a brief gap between jobs in another experiment's queue.

```bash
tmux new-session -d -s mixdiv2k \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0" bash scripts/experiments/setup/nm_mixture_diversity_heldout_cls_2k/run_train_tucker.sh \
   > scripts/experiments/setup/nm_mixture_diversity_heldout_cls_2k/run_logs/orchestrator.log 2>&1'
```

Or leave a persistent waiter when GPU 0 is occupied:

```bash
tmux new-session -d -s mixdiv2k-wait \
  'GPU=0 bash scripts/experiments/setup/nm_mixture_diversity_heldout_cls_2k/wait_and_train_tucker.sh'
```

Completed step-2000 checkpoints are skipped on restart. Filters are available for a
staged launch, for example `TARGETS=twibot20 SIZES=1,2`.

After all training completes:

```bash
DRY_RUN=1 GPUS="0,1" \
  bash scripts/experiments/setup/nm_mixture_diversity_heldout_cls_2k/eval_cls_tucker.sh

GPUS="0,1" \
  bash scripts/experiments/setup/nm_mixture_diversity_heldout_cls_2k/eval_cls_tucker.sh
```

Do not create the matching analysis folder until results exist. The final analysis
should report all 98 points per target, target-wise means and intervals by `k`, the
paired change from `k=1` to each larger `k`, and a target-fixed-effect trend. Because
training uses one seed, composition variation is measured exhaustively but training
seed uncertainty is not; avoid presenting variation across donor combinations as a
seed confidence interval.

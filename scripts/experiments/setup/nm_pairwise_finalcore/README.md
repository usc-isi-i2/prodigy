# All-pairs neighbor-matching sweep

This experiment trains one PRODIGY model on every unordered pair of the nine
final-core social-graph sources, then evaluates every model on each individual
source's frozen NM test stream.

## Registered scope

- 9 sources, hence `9 choose 2 = 36` pair models;
- seed 0 only;
- all 9 test targets, hence 324 model-target cells;
- accuracy, macro F1, and macro one-vs-rest ROC-AUC;
- 512 fixed test episodes per cell; and
- no checkpoint selection: every result uses `state_dict_2500.ckpt`.

The pair sources are balanced at the episode level. Every episode stays inside
one source, and the four episodes in an optimizer batch are independent balanced
draws over the pair. Training positives and message passing use `static_train`;
test positives use the disjoint `static_test` relation. The architecture,
optimizer, two-hop context sampler, 30-way/3-shot/4-query task, batch size 4, and
2,500-update budget are copied from `setup/final_core/training.yaml`.

All 36 models are trained again from random initialization at one revision. The
three pairs already represented as rung 2 in the original final-core ladders are
therefore useful replication controls rather than mixed-revision checkpoint
reuses.

## Fast shared-graph training

The launcher loads the 35M-node all-nine graph once into shared CPU memory and
runs independent trainers against it. It uses all owned Tucker GPUs 0--3, eight
active models per GPU, four loader workers per model, and a total worker budget
of 128. The final four models run as the first wave frees slots.

The previous eight-model validation measured 57.9 aggregate episodes/s on one
H100 after warmup. Each pair model consumes 10,000 episodes, so the sweep contains
360,000 episodes. Linear four-GPU scaling predicts about 26 minutes of steady
training; allowing graph setup, the short final queue, startup, logging, and
checkpoint writes gives a planning ETA of **30--35 minutes**. This is an estimate,
not a four-GPU benchmark. `verify_training.py` reports the realized aggregate
optimizer-step rate when the run finishes.

## Isolated Tucker worktree

Create the worktree only after confirming no job owns it:

```bash
cd /dataMeR1/phil/gfm
git -C prodigy fetch origin codex/nm-pairwise-fast
git -C prodigy worktree add ../prodigy-nm-pairs \
  origin/codex/nm-pairwise-fast
cd /dataMeR1/phil/gfm/prodigy-nm-pairs
git config core.hooksPath .githooks
```

The generated configs are committed. Rebuilding them is deterministic:

```bash
python scripts/experiments/setup/nm_pairwise_finalcore/make_configs.py \
  --replace
```

Check the full resolved plan without touching GPUs or loading the graph:

```bash
DRY_RUN=1 bash scripts/experiments/setup/nm_pairwise_finalcore/run_training_tucker.sh
```

Then validate simultaneous CUDA contexts on GPUs 0--3 without loading the graph:

```bash
PREFLIGHT_ONLY=1 bash scripts/experiments/setup/nm_pairwise_finalcore/run_training_tucker.sh
```

Production belongs in tmux:

```bash
tmux new-session -d -s nmpairs \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   cd /dataMeR1/phil/gfm/prodigy-nm-pairs; \
   bash scripts/experiments/setup/nm_pairwise_finalcore/run_training_tucker.sh \
   > log/nm_pairwise_finalcore_orchestrator.log 2>&1'
```

The launcher refuses busy GPUs, insufficient host RAM or `/dev/shm`, and an
existing output directory. The immutable manifest, effective configs, per-model
logs, checkpoints, and measured throughput live under
`log/nm_pairwise_finalcore/shared_seed0_20260904/`.

## Frozen all-target test

The test queue reuses the published final-core episode fingerprints. Eight
persistent evaluators run two per owned GPU and each loads the graph once. Based
on the completed final-core AUC grid, the expected evaluation time is roughly
**35--40 minutes**, so training plus the full test should take about **65--75
minutes** if Tucker remains idle.

```bash
DRY_RUN=1 bash scripts/experiments/setup/nm_pairwise_finalcore/run_evaluation_tucker.sh

tmux new-session -d -s nmpairs-eval \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   cd /dataMeR1/phil/gfm/prodigy-nm-pairs; \
   bash scripts/experiments/setup/nm_pairwise_finalcore/run_evaluation_tucker.sh \
   > log/nm_pairwise_finalcore_eval_orchestrator.log 2>&1'
```

The queue performs a small eight-worker smoke before the resumable production
grid. Strict aggregation refuses partial, duplicate, wrong-protocol, wrong-stream,
or non-finite cells. Its outputs include the long 324-cell table, three 36x9
metric matrices, a seen-versus-held-out summary, and a completeness receipt under
`log/nm_pairwise_finalcore_eval/production/bs32/summary/`.

# TwiBot-20 transfer

Experiments treating **TwiBot-20** (Twitter bot detection) as a
node-classification domain in the cross-task/cross-dataset transfer study.

The graph is the reconstructed **retweet** graph with bio-embedding node
features (see `data/data/twibot20/README.md`): 162,990 nodes, 2,010,925 directed
edges, 768-d zero-filled bio features, labels `["human", "bot"]` (11,826
labeled; the rest are unlabeled `support` context).

## Phase 1 — Train-on-TwiBot-20 smoke

Validates that TwiBot-20 trains end-to-end through the PRODIGY dataloader as a
binary (bot-vs-human) classification task, and produces a first TwiBot-20 source
checkpoint. Config: [`twibot20_cls_smoke.yaml`](twibot20_cls_smoke.yaml)
(`task_name: classification`, `n_way: 2`, tiny caps, `epochs: 2`).

```bash
# Preview the command:
DRY_RUN=1 bash scripts/experiments/twibot20_transfer/train_twibot20_tucker.sh

# Run the smoke (pick a free GPU first with nvidia-smi):
DEVICE=0 bash scripts/experiments/twibot20_transfer/train_twibot20_tucker.sh
```

Checkpoints land under `state/twibot20_cls_smoke_<timestamp>/`, logs under `log/`.

## Phase 2 — Full train-on-TwiBot-20 run

Real run via [`twibot20_cls.yaml`](twibot20_cls.yaml) (12 epochs x 10k episodes =
120k steps; eval/checkpoint once per epoch). No CLI overrides needed:

```bash
CONFIG_PATH=scripts/experiments/twibot20_transfer/twibot20_cls.yaml \
DEVICE=0 bash scripts/experiments/twibot20_transfer/train_twibot20_tucker.sh
```

`eval_step`/`checkpoint_step` are baked into the YAML at `10000` (= one
`dataset_len_cap` epoch), so eval runs ~12 times over the run. If you change
`dataset_len_cap`, change `eval_step`/`checkpoint_step` to match — a step is one
batch, total steps = `epochs x dataset_len_cap`, and eval fires on
`step % eval_step == 0`. Checkpoints land under `state/twibot20_cls_<timestamp>/`.

## Notes

- Config alignment: `twibot20_cls.yaml` mirrors the merged-graph NM setup
  (`scripts/experiments/nm_transfer_matrix/merged_nm.yaml`) as closely as
  classification allows — `batch_size: 1`, `n_hop: 1`, `n_shots: 3`,
  `n_query: 4`, `dataset_len_cap: 10000`, `val/test_len_cap: 500`,
  `workers: 16`, `epochs: 12`. The one forced difference is `n_way: 2` (capped at
  the 2 labels; merged NM uses `n_way: 30`). Keep `batch_size` at 1 — using 8
  runs 8x the episodes per epoch and makes iterations look very slow.
- Splits: the classification path builds its own stratified node splits
  (seed=0), not TwiBot-20's official `split.csv` (which is preserved on the graph
  as `data.{train,val,test,support}_mask`). This matches the protocol used by the
  other transfer datasets, keeping comparisons apples-to-apples.
- `input_dim` is auto-inferred from the graph's 768-d features.
- Edge features (`n_retweets`) are off by default; add `--use_edge_features true`
  to enable them.

## Experiment (a) — Train NM on TwiBot-20, eval everywhere (transfer OUT)

Train a neighbor-matching model on the TwiBot-20 retweet graph, then evaluate
that checkpoint on every other graph/task. Config
[`twibot20_nm.yaml`](twibot20_nm.yaml) mirrors `nm_transfer_matrix/merged_nm.yaml`
(NM, 30-way, batch_size 1, 12 epochs x 10k = 120k steps).

```bash
# 1. Train (GPU; pick a free device):
CONFIG_PATH=scripts/experiments/twibot20_transfer/twibot20_nm.yaml \
DEVICE=0 bash scripts/experiments/twibot20_transfer/train_twibot20_tucker.sh

# 2. Build the model list from the trained run's final checkpoint:
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state \
  bash scripts/experiments/twibot20_transfer/make_model_list_source.sh

# 3. Eval on all graphs/tasks (NM 30-way 3-shot + classification; LP auto-skipped):
bash scripts/experiments/twibot20_transfer/eval_source_everywhere_tucker.sh
#    add --gpus 0,1,2 to parallelize, --dry-run to preview.
```

## Experiment (b) — Eval merged-strategy models on TwiBot-20 (transfer IN)

Evaluate the checkpoints from the merged-vs-single NM study on TwiBot-20 (NM +
bot-vs-human classification). The 9 models:

- ukr/covid: `nm_matrix_ukr`, `nm_matrix_covid`, `nm_matrix_merged`,
  `nm_xsrc_within_source`
- covid/midterm: `nm_cm_covid`, `nm_cm_midterm`, `nm_cm_merged`, `nm_cm_within`,
  `nm_cm_within_balanced`

```bash
# 1. Build the model list from those runs' final checkpoints:
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state \
  bash scripts/experiments/twibot20_transfer/make_model_list_merged.sh

# 2. Eval them on TwiBot-20:
bash scripts/experiments/twibot20_transfer/eval_merged_on_twibot20_tucker.sh
#    add --gpus 0,1,2 to parallelize, --dry-run to preview.
```

Both experiments require `twibot20` in the shared eval harness `DATASETS`
(`scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py`) — added here.
Results land under `log/eval_*_to_twibot20_*` (b) and `log/eval_nm_twibot20_to_*`
(a); parse them with the same tooling as the other transfer matrices.

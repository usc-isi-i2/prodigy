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

## Next (not yet scaffolded)

- Transfer INTO TwiBot-20: evaluate checkpoints trained on other datasets on
  TwiBot-20 classification (zero- and fixed-adaptation), mirroring
  `scripts/experiments/covid_task_transfer_matrix/`.

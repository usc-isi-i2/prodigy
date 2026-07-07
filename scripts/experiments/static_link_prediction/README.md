# Static Link Prediction benchmark task

Evaluate whether a pretrained model's representations can tell **present from
absent edges** in a retweet graph — an edge-level structural task, without the
temporal split that `temporal_link_prediction` requires.

## Protocol

- **Edge split** (built at graph-construction/enrichment time): existing edges are
  split ~85/15 over *undirected pairs* into a `static_background` view (kept for
  message passing) and a `static_holdout` view (the positive targets). Held-out
  edges are **removed from the background graph**, and the split is over undirected
  pairs, so a held-out `(u,v)` cannot leak via `(v,u)` left in the background.
- **Episodes**: for a sampled center node, positives are its held-out edges;
  negatives are **hard** — nodes two hops away in the background graph but not
  direct neighbours (`--slp-hard-negatives True`, the default). Hard negatives keep
  the task discriminative; random negatives make AUC saturate near 1.0. Set
  `--slp-hard-negatives False` to compare against the easy (random) setting.
- **Encoder input**: `--edge_view static_background` — the model never aggregates
  over the edges it is scored on.

## Episode sizing (sparse graphs)

An episode needs `n_shots + n_query` held-out edges on a **single center node**.
Retweet graphs are power-law/sparse (midterm avg degree ≈ 2.7), so large values
fail with "could not find a center with >= N held-out positive edges". Use
**zero-shot (`--shots 0`) with small `--slp-n-query` (default 4)** so low-degree
centers qualify; denser graphs (e.g. twibot20, avg degree ≈ 12) also support a
few-shot variant. Note this evaluates on sufficiently-connected centers.

## Metrics

Binary scoring of held-out edges vs. hard negatives: **ROC-AUC** (headline, with
hard negatives), plus accuracy and F1 (see `experiments/trainer.py`).

## Datasets

Runs on all five retweet nets (**midterm, ukr_rus_twitter, covid19_twitter,
cp_hk_twitter, twibot20**) — any graph with a `static_holdout` view; the runner
gates on its presence.

## Run

1. **Enrich once** (adds the static views; graph-construction env):

   ```bash
   DATA_ROOT=/dataMeR2/phil/data bash scripts/graph_construction/enrich_all_graphs.sh
   ```

   (Graphs rebuilt from scratch already include the views by default.)

2. **Evaluate** (prodigy env):

   ```bash
   bash scripts/experiments/static_link_prediction/run_static_lp_eval.sh \
     --checkpoint-run-dir /dataMeR2/phil/gfm/prodigy/state/<run> --gpus 0,1
   ```

Results are collected in `scripts/plotting/static_link_prediction/`.

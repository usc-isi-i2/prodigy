# NM cross-source-shortcut test

## Hypothesis

When NM is trained on the **disjoint merged** ukr+covid graph with naive
(proportional) sampling, each episode's `n_way` candidates are drawn from the
whole graph, so a single episode usually contains centers from **both** sources.
The model can then separate the positive (a true neighbor of the center) from the
negatives by exploiting **source-level feature differences** (covid nodes look
different from ukr nodes) — a shortcut — instead of learning fine-grained
**within-source** neighborhood structure.

At test time each domain is evaluated on its own, so every candidate shares a
source and the shortcut is useless — which would explain why the merged model
transfers *worse* than a single-source model.

**Prediction:** if we confine every episode to a single source (no cross-source
negatives), the shortcut disappears, the model must learn within-source structure,
and the merged model should recover toward single-source AUC.

## The one variable changed

This run is byte-for-byte identical to `nm_transfer_matrix/merged_nm.yaml`
(same merged graph, plain defaults, 120k-episode budget, seed 0) **except** one
new flag:

```yaml
neighbor_sampling_episode_source: graph_id
```

It confines each NM episode to one source. The source is chosen **proportional to
its node count**, so the per-node center marginal is *identical* to naive
proportional sampling — the only thing that changes is that an episode's negatives
all share a source. Nothing else differs, so any AUC change is attributable to the
cross-source-negative effect alone.

## Code change (new flag)

Added `--neighbor_sampling_episode_source` (`experiments/params.py`), threaded via
the trainer kwargs (`experiments/trainer.py`), implemented in
`data/dataloader.py::NeighborTask._sample_confined` and wired in the merged-graph
loader (`data/covid19_twitter.py`). Requires per-node `graph.graph_id` (present on
the disjoint merge). When unset, behavior is unchanged.

> Note: while wiring this, the pre-existing `neighbor_sampling_strata` flag (which
> *balances* sources within an episode) was found to never reach the loader; it is
> now threaded through the same path. Default `""` → no behavior change.

## Run (Tucker, prodigy env, in tmux)

```bash
cd scripts/experiments/nm_cross_source_shortcut

# 1. Smoke-test the flag first (1 epoch, tiny) — confirm the strata line prints:
DRY_RUN=0 ./train_tucker.sh --device 3 --epochs 1 -ds_cap 20 -eval_step 20 -ckpt_step 20 \
  --prefix nm_xsrc_smoke
#   look for: "Neighbor sampling graph_id strata (confine-to-one-source): ukr_rus:..., covid:..."

# 2. Full run (GPU 3 is free while the matrix trains on 0/1/2):
./train_tucker.sh

# 3. Eval on both single-source domains, then compare:
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
./eval_tucker.sh
python3 compare_shortcut.py --log-root /dataMeR1/phil/gfm/prodigy/log
```

## Reading the result

`compare_shortcut.py` prints test AUC for four regimes on ukr and covid:

```
regime                      test:ukr     test:covid
single ukr                       ...          ...     <- ceiling for covid? no, for ukr
single covid                     ...          ...
merged proportional              ...          ...     <- baseline (shortcut ON)
merged within-source             ...          ...     <- shortcut OFF
```

and the verdict per cross-domain cell, e.g.
`test covid: proportional=… within-source=… (Δ=…) single-ceiling=… -> shortcut SUPPORTED`.

The hypothesis is supported if **within-source > proportional** on the
cross-domain cells and moves toward the single-source ceiling. The first three
rows are reused from the `nm_transfer_matrix` eval logs, so only the
within-source model needs fresh training + eval.
```

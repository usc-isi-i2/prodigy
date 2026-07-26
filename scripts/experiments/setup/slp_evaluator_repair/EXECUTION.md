# Gate 0 — repair the link-prediction evaluator, then rescore

**Status:** evaluator built and validated offline; nothing has been run on Tucker.

## Why

The episodic `static_link_prediction` eval cannot measure link prediction. Three
independent defects (all verified in code, see commit `8efb630`):

1. **Center-blind.** `StaticLinkTask.sample` returns `{(0, center): neg, (1, center): pos}`
   ([data/midterm.py:181](../../../../data/midterm.py)). `center` is only the second element of
   the label-map *key*; the encoded subgraphs come from the candidate lists. The score was
   `f(v)` — the queried edge's other endpoint never entered the model input.
2. **Frozen random class prototypes.** `ignore_label_embeddings` defaults True, so label
   reps are `nn.Embedding` rows, frozen during pretraining. `--shots 0` also sets
   `--zero_shot True`, under which `forward_metagraph` skips message passing entirely — so
   no support example could inform them either. With `final_label_mlp = Identity`, the two
   "edge / no-edge" prototypes are literally fixed random vectors.
3. **Degree-confounded negatives.** Positives are drawn from a center's holdout neighbours
   (holdout-degree ≥ 1 by construction); negatives carry no such condition.

This explains the anomalies previously logged as curiosities: controls emitting constant
predictors, recurring sub-chance AUCs, and rewire-kills-it / permute-is-a-no-op (degree
*is* topology, so the 2×2 ablation could not separate a degree shortcut from pairwise
adjacency).

**Unaffected:** classification (10-shot, `zero_shot=False`, so prototypes are real) and
regression (bypasses the decoder via `regression_head`). Those numbers stand.

## Step 0 — offline gates (no cluster, no GPU)

```bash
python scripts/eval/pair_link_eval.py --self-test && python scripts/eval/tests/test_embed_walk.py
```

23 protocol checks + 5 encoder-walk checks. Both green as of `0825ff2`.

## Step 1 — Tucker smoke test (do this before any sweep)

Encoder reconstruction is the one piece that cannot be validated off-cluster. This loads a
real checkpoint, asserts no encoder weight is missing, and embeds a few hundred nodes.
**Check GPU occupancy first** — another session may be running.

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

```bash
cd /dataMeR1/phil/gfm/prodigy-slpfix && LD_LIBRARY_PATH=/home/mhchu/miniconda3/envs/prodigy/lib /home/mhchu/miniconda3/envs/prodigy/bin/python scripts/eval/tests/smoke_test_tucker.py --graph /dataMeR1/phil/data/covid19_twitter/graphs/retweet_graph_parquet.pt --checkpoint /dataMeR1/phil/gfm/prodigy-mtr/state/mtr_MIX_*/checkpoint/state_dict_30000.ckpt --device cuda
```

If it reports missing encoder weights, the `--emb-dim/--gnn-type/--n-layer/--layers` flags
do not match how that run was trained — read the run's config rather than forcing it.

## Step 2 — rescore one checkpoint on one graph

```bash
cd /dataMeR1/phil/gfm/prodigy-slpfix && LD_LIBRARY_PATH=/home/mhchu/miniconda3/envs/prodigy/lib /home/mhchu/miniconda3/envs/prodigy/bin/python scripts/eval/pair_link_ckpt.py --graph /dataMeR1/phil/data/covid19_twitter/graphs/retweet_graph_parquet.pt --checkpoint /dataMeR1/phil/gfm/prodigy-mtr/state/mtr_MIX_*/checkpoint/state_dict_30000.ckpt --model-name mtr_MIX --out results/mtr_MIX__covid19.json --device cuda
```

Reads to make before trusting anything: `leakage=0`, `sensitivity≈1.000`,
`perm_auc≈0.5`. Any deviation voids the row.

## Step 3 — the rescoring matrix

No retraining. All checkpoints already exist:

| group | checkpoints | location |
|---|---|---|
| rotation | `mtr_{NM,CL,FP,MIX}` | `/dataMeR1/phil/gfm/prodigy-mtr/state/` |
| pairs | `mtp_{NMCL,NMFP,CLFP}` | `/dataMeR1/phil/gfm/prodigy-mtp/state/` |
| corpora | `msc_{cov,all8}_{NM,CL,FP,MIX}` | `/dataMeR1/phil/gfm/prodigy-msc/state/` |

Datasets: the focused-5 (`midterm`, `ukr_rus_twitter`, `covid19_twitter`, `cp_hk_twitter`,
`twibot20`) — any graph carrying `static_holdout`.

Headline condition is `degree_matched`. `random` is the easy control. **`hard_2hop` is
punishing by construction** — a 2-hop negative shares a neighbour with `u`, and in a
community-structured graph sits in `u`'s own block, so it neutralises both community
identity and common-neighbour (the synthetic gate shows CN dropping to 0.525 there).
Report it, but do not read a low number as encoder failure.

## What the result decides

Every arm is reported against CN / Adamic-Adar / preferential-attachment / Jaccard /
raw-feature-cosine floors on the *same* pair set.

- **MIX clears the heuristic floors** → the emergent-LP finding survives on a valid metric,
  and the joint-loss experiment has a real bar to beat.
- **MIX at or below the floors** → the 0.76 was an artifact of the three defects, the
  topology channel needs rebuilding on other evidence, and the joint-loss arm should not be
  judged on LP at all.

Either way this costs no pretraining, and it gates the objective-combination work.

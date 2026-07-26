# multitask_ssl_corpora — does the MIX emergent-sLP contrast survive a corpus change?

Replication of `multitask_ssl_rotation` on **two new pretraining corpora**. The
original run (merged 3-way ukr+covid+midterm retweet corpus) found that MIX — the
per-episode nm/cl/fp rotation — is the **only** arm with above-chance 0-shot
static link prediction (AUC 0.759 vs <= 0.467 for NM/CL/FP). This experiment asks
whether that contrast is a property of the *objective rotation* (should reproduce
on any corpus) or an accident of the 3-way corpus.

## Design

2 corpora x 4 arms (NM, CL, FP, MIX — no pairs), **everything else identical** to
`../multitask_ssl_rotation/configs/*.yaml`: seed 0, epochs 4 x dataset_len_cap
10000 = 40k episodes, batch_size 1, bio-only 768-d GTE features, 1-layer
mean-agg SAGE (emb 256), 30-way/3-shot/4-query episodes, GLOBAL episode sampling,
checkpoint_step 10000. Per config, ONLY dataset/root/graph_filename/prefix/tags
differ from the originals.

| corpus | dataset key | graph |
|---|---|---|
| `cov`  | `covid19_twitter` | `/dataMeR1/phil/data/covid19_twitter/graphs/retweet_graph_parquet.pt` (single source) |
| `all8` | `covid19_twitter` (shared merged-graph loader path) | `/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_all8_retweet_graph.pt` (8 sources) |

Runs are named `msc_<corpus>_<ARM>` (8 total: msc_cov_NM ... msc_all8_MIX).

**Checkpoint off-by-one (kept on purpose):** the trainer runs steps
0..epochs*cap-1 and never checkpoints the final step, so epochs:4 saves ckpts at
10k/20k/**30k** and no 40k ckpt exists. The original arms were all evaluated at
30k; keeping epochs:4 preserves matched-at-30k parity. Do NOT bump to epochs:5.

## Hypotheses

- H1 (objective-driven): on each corpus, MIX shows above-chance 0-shot sLP while
  NM/CL/FP sit at chance — the rotation itself teaches adjacency.
- H2 (corpus-driven): the contrast weakens or vanishes on cov (single-source) or
  all8 (8-source) — the 3-way corpus was load-bearing.
- Secondary: does MIX stay the generalist (near-best cls, competitive reg) per
  corpus? Does corpus breadth (1 vs 3 vs 8 sources) modulate any of it?

## Files

- `configs/{cov,all8}/{NM,CL,FP,MIX}.yaml` — the 8 arm configs.
- `launch_all_tucker.sh` — detached-tmux launch of all 8, 2 per GPU (0-3).
- `make_model_list.sh` — `model_list.txt`, keyed `<corpus>_<ARM>`, 30k ckpts only.
- `run_eval_sweep.sh` — frozen-encoder benchmark (reg 10-shot / slp 0-shot /
  pl 10-shot over midterm, ukr_rus_twitter, covid19_twitter, twibot20,
  election2020) — identical tasks/datasets to the rotation sweep.
- `watch_and_eval.sh` — waits for all 8 x 30k ckpts + session exits, then runs
  model-list + eval sweep + commits result CSVs. Log ends "ALL COMPLETE".
- `EXECUTION.md` — the exact commands as run.

Results land in `scripts/experiments/analysis/{node_regression,
static_link_prediction,node_classification}/data/*.csv` keyed by
model = `<corpus>_<ARM>`; the reading goes to
`scripts/experiments/analysis/multitask_ssl_corpora/FINDINGS.md`
(**interpretation embargoed** until the parallel `mix_slp_ablation` verdict).

## Eval notes (known, intended)

- Eval episodes are seeded by split name, not `--seed` — identical eval episodes
  across arms is intended and matches the original run.
- The shared eval runner hardcodes NM n_way=3 for its NM probe — known/fine.
- All 8 arms are bio-768/mean-SAGE: no STRUCTURAL/GNN_TYPE args needed.

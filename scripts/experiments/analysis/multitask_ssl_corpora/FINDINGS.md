# multitask_ssl_corpora — findings

> **RESULTS PENDING — interpretation embargoed until the `mix_slp_ablation`
> verdict.** A parallel eval-time ablation (Workstream A) is determining whether
> the original MIX static-LP finding is topological or a feature artifact. Until
> that verdict is in, this file records design and hypotheses only. Numbers and
> tables may be added by a later session; NO interpretive claims before the
> embargo lifts.

## What was run

Replication of `multitask_ssl_rotation` (4 arms: NM, CL, FP, MIX) on two new
pretraining corpora, everything else identical (seed 0, 40k-episode budget,
bio-768/mean-SAGE, global sampling; all arms evaluated at the 30k checkpoint —
the trainer's off-by-one means 30k is the terminal ckpt, matching the original):

- **cov** — covid-only single-source retweet graph
  (`covid19_twitter/graphs/retweet_graph_parquet.pt`)
- **all8** — merged 8-source retweet graph
  (`merged/graphs/ukr_rus_covid_midterm_all8_retweet_graph.pt`)

Setup, configs, and exact commands: `../../setup/multitask_ssl_corpora/`.
Original result being replicated: `../multitask_ssl_rotation/FINDINGS.md`
(MIX the only arm with above-chance 0-shot static-LP, 0.759 vs <= 0.467).

## Hypotheses (pre-registered, before results)

1. **Objective-driven (H1):** on each new corpus, MIX shows above-chance 0-shot
   static-LP while NM/CL/FP sit at chance — the 3-way-synergy rotation itself
   teaches adjacency, corpus-independently.
2. **Corpus-driven (H2):** the contrast weakens/vanishes on cov (single source)
   and/or all8 (8 sources) — corpus composition was load-bearing.
3. Secondary: MIX remains the generalist (near-best classification, competitive
   regression) per corpus; corpus breadth (1 vs 3 vs 8 sources) may modulate
   absolute levels without changing the MIX-vs-controls ordering.

## Results

_PENDING — to be filled from
`scripts/experiments/analysis/{node_regression,static_link_prediction,node_classification}/data/*.csv`
rows with model in {cov,all8}_{NM,CL,FP,MIX} once the watcher's eval sweep
completes (`/tmp/msc_watcher.log` ends with "ALL COMPLETE")._

## Interpretation

_EMBARGOED until the `mix_slp_ablation` verdict (Workstream A)._

# multitask_ssl_corpora — findings

**Headline: the emergent-MIX static-LP contrast does NOT replicate on either new
corpus — corpus composition, not the objective rotation alone, is load-bearing.**
On covid-only, three arms (CL, FP, MIX) sit modestly above chance with no
MIX-unique advantage; on the all8 merge the pattern *inverts* — NM is the best
sLP arm (0.62–0.74) while MIX and CL fall *below* chance. The parallel
`mix_slp_ablation` (Workstream A) confirmed the ORIGINAL 3-way MIX sLP signal is
genuinely topological (dies under degree-preserving rewiring, unaffected by
feature permutation), so the original finding stands — but as a
corpus-x-objective interaction, not a corpus-general property of the rotation.

## What was run

Replication of `multitask_ssl_rotation` (4 arms: NM, CL, FP, MIX) on two new
pretraining corpora, everything else identical (seed 0, 40k-episode budget =
epochs 4 x cap 10000, batch_size 1, bio-768/mean-SAGE, 30-way/3-shot/4-query,
global sampling; all arms evaluated at the 30k checkpoint — the trainer's
off-by-one means 30k is the terminal ckpt, matching the original):

- **cov** — covid-only single-source retweet graph
  (`covid19_twitter/graphs/retweet_graph_parquet.pt`)
- **all8** — merged 8-source retweet graph
  (`merged/graphs/ukr_rus_covid_midterm_all8_retweet_graph.pt`)

Setup/configs/commands: `../../setup/multitask_ssl_corpora/`. Original result:
`../multitask_ssl_rotation/FINDINGS.md` (MIX the only arm with above-chance
0-shot static-LP on the 3-way corpus: 0.759 vs <= 0.467 for NM/CL/FP).
Frozen-encoder eval sweep identical to the original (focused-5 datasets;
eval episodes seeded by split name => identical episodes across arms). Evidence
CSVs: `data/{static_link_prediction,node_classification,node_regression}.csv`
(model key = `<corpus>_<ARM>`); rows also appended to the shared plotting CSVs.

## Pre-registered hypotheses

1. **H1 objective-driven:** MIX above-chance sLP / controls at chance on each
   corpus. 2. **H2 corpus-driven:** the contrast weakens or vanishes off the
   3-way corpus. 3. Secondary: MIX stays the generalist.

**Verdict: H2.** (With the twist that all8 flips the ordering entirely.)

## Results (test split, 30k ckpts, single seed)

### Static link prediction, 0-shot ROC-AUC (chance = 0.5)

| model | covid19 | midterm | twibot20 | ukr_rus | vs original (3-way) |
|---|---|---|---|---|---|
| cov_NM   | 0.196 | 0.303 | 0.222 | 0.143 | orig NM 0.42–0.47 |
| cov_CL   | 0.669 | 0.638 | 0.604 | 0.644 | orig CL at/below chance |
| cov_FP   | 0.579 | 0.627 | 0.537 | 0.600 | orig FP at/below chance |
| cov_MIX  | 0.545 | 0.510 | 0.617 | 0.588 | orig MIX 0.759 |
| all8_NM  | 0.737 | 0.621 | 0.742 | 0.665 | |
| all8_CL  | 0.237 | 0.353 | 0.203 | 0.181 | |
| all8_FP  | 0.549 | 0.558 | 0.555 | 0.569 | |
| all8_MIX | 0.272 | 0.410 | 0.231 | 0.295 | |

- **cov:** no MIX-unique signal. CL is the best sLP arm (0.60–0.67), FP and MIX
  modestly above chance, NM strongly BELOW chance (0.14–0.30 — an inverted,
  i.e. anti-correlated, embedding-distance signal, not noise).
- **all8:** inversion of the original — NM clears chance everywhere (up to
  0.74), FP is marginal, MIX and CL are far below chance.
- Sub-chance AUCs recur across arms/corpora: encoders often learn a *consistent*
  adjacency signal with the wrong sign for the frozen-probe scoring. Any
  mechanism story must explain sign flips, not just magnitude.

### Node classification, 10-shot ROC-AUC

| model | election2020 | twibot20 |
|---|---|---|
| cov_NM | 0.979 | 0.625 |
| cov_CL | 0.577 | 0.635 |
| cov_FP | 0.310 | 0.483 |
| cov_MIX | 0.980 | 0.621 |
| all8_NM | 0.980 | 0.646 |
| all8_CL | 0.650 | 0.646 |
| all8_FP | 0.626 | 0.588 |
| all8_MIX | 0.978 | 0.643 |

NM and MIX are near-ceiling on election2020 and best-or-tied on twibot20 in
both corpora — the "NM/MIX good at classification" half of the original DOES
replicate, corpus-independently.

### Node regression, 10-shot Spearman (mean over 3 targets)

| model | covid19 | midterm | twibot20 | ukr_rus |
|---|---|---|---|---|
| cov_NM | -0.001 | -0.063 | -0.006 | -0.033 |
| cov_CL | -0.075 | 0.035 | -0.026 | 0.001 |
| cov_FP | 0.089 | 0.115 | -0.021 | 0.131 |
| cov_MIX | 0.017 | -0.153 | 0.054 | -0.039 |
| all8_NM | -0.010 | 0.113 | 0.122 | 0.001 |
| all8_CL | -0.071 | -0.080 | 0.074 | -0.032 |
| all8_FP | 0.025 | 0.060 | -0.039 | 0.081 |
| all8_MIX | 0.041 | 0.153 | 0.025 | 0.091 |

All weak (|rho| <= 0.15, mixed signs) — regression is near-noise at this budget
on both corpora, as it was for most arms originally (FP slightly best on cov,
matching FP's feature-reconstruction bias).

## Interpretation

Interpretation was embargoed until the `mix_slp_ablation` verdict; that verdict
is now in (2026-07-21): the original 3-way MIX sLP signal is **topological**
(collapses to ~0.53–0.64 under degree-preserving edge rewiring, unaffected by
feature permutation; NM at chance in all conditions; unmodified anchor
reproduced the original numbers). Reading the replication in that light:

1. **The rotation does not intrinsically teach adjacency.** If 3-way-rotation =>
   emergent sLP were objective-driven, it should have appeared on cov and all8.
   It appeared on neither; on all8 MIX is among the worst sLP arms.
2. **The original finding still stands, but is narrower than hoped:** on the
   3-way corpus MIX genuinely encodes real topology (per A), yet that emergence
   is an interaction with corpus composition. Candidate moderators for follow-up:
   number/balance of sources (3 vs 1 vs 8), per-source episode share under
   global sampling, and graph-scale/density differences (cov and all8 graphs are
   ~10–100x the 3-way corpus by bytes).
3. **Which-objective-wins-sLP is itself corpus-dependent** (CL on cov, NM on
   all8, MIX on 3-way) and frequently sign-flipped. Frozen-probe 0-shot sLP on
   these encoders is a fragile capability readout; multi-seed replication and a
   sign-aware probe would be needed before any per-objective claim.
4. The stable, corpus-general result of the whole program remains: **NM/MIX
   dominate few-shot classification** (near-ceiling election2020, best-tied
   twibot20 on all three corpora), and regression stays near-noise.

Caveats: single seed per arm (matching the original); 2 runs/GPU contention
(GPU 3 shared with an unrelated ollama server); all readings at the 30k ckpt.

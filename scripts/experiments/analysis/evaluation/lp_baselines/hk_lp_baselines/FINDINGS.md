# HK direct-link baseline pilot — 2026-09-08

The frozen HK PRODIGY/NM encoder has a small ROC-AUC advantage over raw-feature
cosine on this test, but trails both LP-trained baselines substantially. Its
average precision is slightly below raw-feature cosine. This does **not** show
that PRODIGY improves LP over a trained MLP or GraphSAGE.

| Method | Validation AUC | Test AUC | Test AP |
|---|---:|---:|---:|
| Raw node-feature cosine | 0.5305 | 0.5364 | 0.5339 |
| Raw center + mean neighbors, cosine | 0.5316 | 0.5387 | 0.5298 |
| LP-trained MLP | 0.6794 | 0.6762 | 0.6430 |
| LP-trained one-hop GraphSAGE | 0.6825 | 0.6849 | 0.6465 |
| Frozen PRODIGY/NM encoder, cosine | 0.5524 | 0.5550 | 0.5299 |
| Common neighbors | 0.5390 | 0.5431 | 0.5165 |
| Adamic-Adar | 0.5375 | 0.5438 | 0.5158 |
| Jaccard | 0.5282 | 0.5363 | 0.5068 |
| Preferential attachment | 0.5052 | 0.5052 | 0.5045 |

All methods use identical saved held-out pairs. Each validation/test set has
2,000 positive edges and 2,000 matched nonedges. Higher is better; chance AUC
and constant-score AP are 0.5 on these balanced sets. Validation selected the
positive score orientation for every method. No test tuning or checkpoint
selection occurred. These numbers must not be mixed with historical LP tables
using different pair sets or negative sampling.

## Protocol and limits

- HK has 333,800 nodes with 768-dimensional profile features. The existing
  canonical 70/15/15 edge views were reused. Training, neighborhood aggregation,
  and the original frozen checkpoint's NM pretraining use `static_train`.
- Each nonedge fixes a randomly chosen endpoint of its matched positive and
  replaces the other within the same log2 training-degree bin; isolates are
  separate. Negatives exclude every recorded edge in all three splits, and
  all train/validation/test unordered pair sets are disjoint.
- Bounded negative search rejected 50/2,050 validation and 60/2,060 test positive
  proposals. Thus the population is sampled edges admitting the specified
  matched corruption, not all edges or all possible node pairs. Sparse graphs
  can also contain unobserved true links; labels mean recorded edge/nonedge.
- The MLP and GraphSAGE each trained for 2,500 updates on the same 100,000 training
  edges and 100,000 nonedges, using identical balanced batches and optimization.
  Both learn a shared node embedding scored by cosine. GraphSAGE is a one-hop
  mean convolution followed by a projection; the MLP has two dense layers.
- PRODIGY is the existing HK seed-0 NM checkpoint at step 2,500, used without LP
  adaptation. Its own two-hop [9,9] sampled directed contexts differ from the
  one-hop full undirected means used by the baselines. Training objective,
  examples per update, total exposure, and compute are not matched to PRODIGY.
- This evaluates **encoder-only direct LP**. There are no NM episodes, support
  labels, or metagraph predictions. It does not isolate an architecture effect,
  explain NM failures, or measure the full in-context PRODIGY system.
- One graph, training seed and sampled pair set: the gaps are pilot point
  estimates, not replicated benchmark improvements. The 0.009 GraphSAGE–MLP
  difference does not establish a robust winner between those baselines.

## Feature availability

At least one endpoint has all-zero raw features in 1,129/4,000 test pairs (28.2%).
The predeclared feature-availability strata show:

| Test subset | Pairs (positive / negative) | Raw cosine AUC | MLP AUC | GraphSAGE AUC | PRODIGY AUC |
|---|---:|---:|---:|---:|---:|
| Both endpoints have features | 2,871 (1,438 / 1,433) | 0.5682 | 0.7024 | 0.6908 | 0.5768 |
| At least one zero-feature endpoint | 1,129 (562 / 567) | 0.5000 | 0.6063 | 0.6689 | 0.4936 |

GraphSAGE's advantage over the MLP is concentrated in the zero-feature subset;
the MLP is stronger when both endpoints have features. This is a descriptive
subgroup comparison, not a controlled feature-removal intervention. Simply
adding neighbor means yields little overall benefit. The useful performance
here comes from the LP-trained models; this comparison cannot distinguish
objective mismatch from architecture or training exposure as the cause of
PRODIGY's weaker transfer.

## Validation and artifacts

Protocol tests check split disjointness, true-edge rejection, degree matching,
isolates, and tie-aware AP. A synthetic gate verifies the precomputed mean
GraphSAGE against actual PyG message passing. The frozen PRODIGY weights loaded
strictly and remained unchanged. Endpoint permutation reduces MLP, GraphSAGE,
and PRODIGY AUCs to 0.4953, 0.4917, and 0.4906, respectively.

Aggregate evidence: [data/results.json](data/results.json).
Reproduction: [setup README](../../../../setup/hk_lp_baselines/README.md).
The [cached verifier](verify_cached.py) checks original training-feature
identity, all saved context edges, pair disjointness, and score/metric replay
without graph sampling or model encoding.
It [passed](data/cache_verification.json): all 333,800 source feature rows match,
all 373,241 saved neighborhood edge occurrences belong to training, and cached
score replay error is exactly zero. This separate read-only verification took
36.1 seconds.

Private pair IDs, complete sampled contexts, embeddings, checkpoints, training
losses, scores and receipts remain on Tucker under
`/dataMeR1/phil/gfm/lp_baselines/hk_20260908`.
The run exited successfully at code revision `b19a7a6e`.

Execution took 27.3 seconds: 6.0 s hashing/loading/pair generation, 0.6 s neighbor
precomputation, 5.1 s MLP training/export, 5.1 s GraphSAGE training/export,
3.9 s PRODIGY sampling/encoding/cache, and 4.5 s scoring/output. Remaining time
is bookkeeping. Human/agent setup and validation are separate from this compute
time. A stale configuration path was fixed during a two-step smoke check before
the full pilot; smoke outputs are not evidence for the table above.

Worktree: `/private/tmp/prodigy-hk-lp-baselines`; branch:
`codex/hk-lp-baselines-20260908`. Tucker used its own matching worktree at
`/dataMeR1/phil/gfm/prodigy-hk-lp-baselines-20260908`, GPU 0. The parent thread's
working tree and other jobs were not modified.

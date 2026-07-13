# feature_ablation — Findings

**Question.** We pretrain mostly on Neighbor Matching (NM). Does NM (and do the
downstream tasks) actually use the node features (GTE bio embeddings), or are they
solved from graph topology? And if the model ignores feature content, could forcing
feature use help? This gates the pretraining-objective direction.

> **Correction up front.** An intermediate conclusion — "NM ignores feature content;
> the encoder strips it" — was **wrong**. It rested on `permute`-invariance alone.
> Adding the `noise` ablation overturned it: **NM relies heavily on real feature
> content.** The doc below is the corrected account; the mistake is called out where
> it happened as a methods lesson.

## Setup

- **Checkpoint:** `nm_matrix_covid` (single-source covid NM, `state_dict_50000.ckpt`).
- **Eval-time ablations** (perturb the model's input, episodes still sampled from the
  real graph so labels stay valid), on two axes — node *distinctness* and real
  *neighborhood content*:

  | condition | node distinct? | node↔feat binding | real neighborhood content |
  |---|:--:|:--:|:--:|
  | intact  | ✓ | correct | ✓ |
  | permute | ✓ | scrambled | ✓ (preserved — shuffled *within* subgraph) |
  | noise   | ✓ | n/a | ✗ (resampled from full-graph distribution) |
  | zero    | ✗ | — | ✗ |

- **Regime:** NM 30-way / 3-shot / test; classification linear-probe 20-shot.
- **Branch:** `experiment/feature-ablation`. Data on Tucker, isolated worktree at
  `/dataMeR1/phil/gfm/prodigy-featabl`.

## Headline

**NM matches a query to a center via their shared neighborhood feature *content*,
not topology and not mere node-distinctness.** Destroying real content (`noise`)
collapses NM to chance exactly like deleting features (`zero`); scrambling only the
binding (`permute`) is harmless. Topology is left intact in both `zero` and `noise`
yet both give chance — and at the deployed `n_hop = 1` there is essentially no
topology to use anyway (star subgraphs; see Findings 5).

## NM ablation (accuracy / ROC-AUC; chance = 1/30 ≈ 0.033)

| dataset | intact | zero | permute | **noise** |
|---|---|---|---|---|
| covid19_twitter (in-domain) | 0.664 / 0.982 | 0.073 / 0.678 | 0.626 / 0.976 | **0.061 / 0.622** |
| midterm | 0.313 / 0.884 | 0.086 / 0.718 | 0.311 / 0.885 | **0.064 / 0.646** |
| twibot20 | 0.406 / 0.924 | 0.066 / 0.671 | 0.407 / 0.927 | **0.058 / 0.632** |

`noise ≈ zero ≈ chance  ≪  permute ≈ intact`, consistently across all three graphs.

## Findings

1. **NM uses real feature content.** `noise` keeps nodes distinct and in-distribution
   but replaces the neighborhood's real bios — and NM collapses to chance. So
   distinctness is not sufficient; the model needs the *real* neighborhood content.
2. **Not the node↔feature binding.** `permute` preserves the subgraph's feature
   multiset (reshuffled among its nodes) and is harmless → the model uses content as a
   permutation-invariant *bag*, not bound to specific nodes.
3. **The features are good.** Raw feature → label logistic regression (no graph):
   AUC 0.95 (election2020), 0.91 (covid_political), 0.71 (twibot20), 0.60
   (ukr_rus_suspended). Not a broken-feature story.
4. **Neighborhoods carry community-level feature signal.** Prototype-NN NM in raw
   feature space (no model): twibot20 0.169 (AUC 0.66) vs permuted control 0.035;
   midterm 0.103 (AUC 0.61) vs 0.032. Weaker than the full model (0.41 / 0.31), but
   real — exactly the signal the model exploits.
5. **"Topology" is definitional at n_hop = 1.** Subgraphs are 1-hop ego-nets (stars);
   the pooling supernode attaches to the center only, so information flows
   neighbors → (data edges) → center → supernode. The edges' only job is to deliver
   the neighbor feature set to the readout — there is no multi-hop structure to use
   independently of features. So "topology barely matters" is architectural, not a
   claim that the model ignores available structure. A clean structure test needs
   **multi-hop retraining** (evaluating the 1-hop checkpoint at 2-hop is just shift).
6. **The methods lesson.** `permute`-invariance does **not** imply content-disuse,
   because permute preserves the content bag. Never conclude feature-disuse from
   permute alone — pair it with `noise`. The earlier Phase-B "encoder strips content"
   claim was permute-only and is therefore **not established** (see below).

## Phase B (classification linear-probe on the frozen representation) — permute-only

AUC intact / zero / permute: election2020 0.979/0.503/0.978, covid_political
0.912/0.613/0.911, twibot20 0.680/0.715/0.673. Permute-invariant on every graph —
but by Finding 6 this does **not** show the encoder discards content. The twibot20
gap (rep 0.680 < raw-feature logreg 0.707) is a *tentative* hint that individual-node
semantics may be under-encoded relative to the neighborhood-aggregate the model uses
for NM — **unconfirmed**, pending a `noise` re-run.

## Implication (revised)

Features are good and NM already **uses** feature content heavily. The original
motivation ("NM ignores features → add a feature-content task") is **overturned**;
if anything, topology is the underused channel — and at 1-hop there is little of it.
The remaining open question is narrower: does the encoder preserve *individual-node*
bio semantics (needed for low-homophily downstream like twibot20), or only the
*neighborhood-aggregate* signature it uses for NM?

## Caveats

- **One checkpoint (covid NM), one seed.** Effect sizes are large and consistent
  across 3 graphs, so robust *within* this checkpoint; cross-checkpoint unverified.
- **twibot20 is the clean cell** (label-homophily 0.40); the political graphs are
  homophily-confounded for the downstream comparison.
- **Phase B is permute-only** — its content claim is not established.
- **n_hop = 1** — the topology conclusion is specific to this setup.

## Wiring (implemented)

- `data/augment.py`: `AblateAllFeatures({zero,permute})` (tokens `FZ`/`FP`),
  `AblateEdges({rewire})` (token `ER`); `noise` reuses `RandomNodeAttr` (`NR1.0`,
  full-graph resample). Unit tests in `data/tests/test_ablate_features.py`.
- `experiments/params.py`: `--ablate_features {none,zero,permute,noise}` composes the
  token into `--augmentation` + forces `--augment_test True`.
- `eval_ckpts_all_graph_tasks_tucker.py`: `--ablate-features` pass-through; ablated
  runs tagged `_ablZ/_ablP/_ablN` so they don't collide.
- Probes: `feature_label_probe.py` (feature→label logreg),
  `feature_only_nm_probe.py` (prototype-NN NM). Parser: `parse_feature_ablation.py`.
- Results: `feature_ablation_results.csv`, `feature_label_probe_results.csv`,
  `feature_only_nm_results.csv`.

## Next

1. **`noise` on the Phase-B classification probe** — settle whether the encoder
   discards individual-node content or the political-graph numbers were homophily.
2. **Feature-reconstruction probe** — predict a node's raw bio from its frozen
   representation (encoded-but-unused vs. not-encoded).
3. **Multi-hop (n_hop = 2) NM training** — the only clean way to ask whether real
   graph structure would add anything beyond the neighbor feature set.

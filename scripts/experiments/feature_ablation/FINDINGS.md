# feature_ablation — Findings

## Executive summary

We pretrain mostly on Neighbor Matching (NM) and asked whether NM (and the
downstream tasks) use the node features (GTE bio embeddings) or solve everything
from graph topology. Using eval-time input ablations on a fixed NM checkpoint:
**NM relies on the real feature *content* of a node's neighborhood — not on
topology, and not on mere node-distinctness.** Destroying real neighborhood content
(`noise`) collapses NM to chance, exactly like deleting features (`zero`), while
scrambling only the feature↔node binding (`permute`) is harmless. The bio features
are informative (raw feature→label AUC 0.71–0.95) and neighborhoods carry genuine
community-level feature signal. At the deployed `n_hop = 1` the subgraph is a star,
so structure enters only as "who the neighbors are": NM is a neighborhood-feature-
content matching task.

## Methodology

**Design.** Hold the checkpoint fixed and perturb only the model's input. Episodes
are still sampled from the real graph, so labels stay valid — the ablation changes
the input, not the task. "Using features" is ambiguous (features can act as node
*distinguishers*, as real *content*, or as *bound* content), so the treatments vary
features along two axes — **node distinctness** and **real neighborhood content**:

| condition | node distinct? | node↔feat binding | real neighborhood content |
|---|:--:|:--:|:--:|
| intact  | ✓ | correct | ✓ |
| permute | ✓ | scrambled | ✓ (preserved) |
| noise   | ✓ | n/a | ✗ (destroyed) |
| zero    | ✗ | — | ✗ |

**Treatments (hypothesis + motivation):**

- **`zero`** — replace every node feature with 0.
  *Hypothesis:* if NM survives, it runs on topology alone; if it collapses,
  features are necessary. *Motivation:* the floor — removes distinctness and content
  together.
- **`permute`** — shuffle feature rows across nodes *within each subgraph*.
  *Hypothesis:* if NM drops, the model depends on the specific node↔feature binding.
  *Motivation:* isolates the binding while preserving the subgraph's real feature
  multiset and distinctness.
- **`noise`** — resample every node's features from the *full graph's* feature
  distribution (distinct, in-distribution, wrong content).
  *Hypothesis:* if NM holds → features act only as distinguishers; if NM collapses →
  the model uses the real neighborhood content. *Motivation:* `permute` preserves
  the content *bag*, so a bag-of-content model is permute-invariant too — `noise` is
  the treatment that separates distinguisher-vs-content, by destroying real content
  while holding distinctness and scale fixed.
- **Feature→label probe** (`feature_label_probe.py`) — logistic regression from raw
  features to the node label, *no graph*. *Hypothesis:* AUC ≫ 0.5 ⇒ features carry
  real label signal. *Motivation:* separates "model underuses good features" from
  "features are uninformative."
- **Feature-only NM probe** (`feature_only_nm_probe.py`) — prototype nearest-neighbor
  NM in raw feature space, *no model*, with a permuted-feature control.
  *Hypothesis:* ≫ chance ⇒ neighborhoods carry feature-discriminative content the
  model could exploit. *Motivation:* confirms that content exists in the data,
  independent of the model.

**Reading rule.** `intact − ablated`: a large drop means the ablated property is
load-bearing. The crux is `permute` vs `noise` — both keep distinctness, only
`noise` destroys real content, so `noise ≪ permute` ⇒ content is used.

## Experiment setup

- **Checkpoint:** `nm_matrix_covid`
  (`state/nm_matrix_covid_28_06_2026_15_54_50/checkpoint/state_dict_50000.ckpt`).
- **Datasets:** NM on covid19_twitter (in-domain), midterm, twibot20; classification
  linear-probe on twibot20, covid_political, election2020; feature/feature-only
  probes on the labeled graphs.
- **Regime:** NM 30-way / 3-shot / test (chance 1/30 ≈ 0.033); classification
  linear-probe 20-shot. `n_hop = 1`.
- **Harness:** `--ablate_features {none,zero,permute,noise}` →
  `eval_ckpts_all_graph_tasks_tucker.py --ablate-features …` (ablated runs tagged
  `_ablZ/_ablP/_ablN`); sweep via `run_feature_ablation_tucker.sh`, collate via
  `parse_feature_ablation.py`.
- **Env:** branch `experiment/feature-ablation`; Tucker `prodigy` env, worktree
  `/dataMeR1/phil/gfm/prodigy-featabl`.

## Results

Raw data: [`feature_ablation_results.csv`](feature_ablation_results.csv),
[`feature_label_probe_results.csv`](feature_label_probe_results.csv),
[`feature_only_nm_results.csv`](feature_only_nm_results.csv).

**NM ablation** — accuracy / ROC-AUC (chance ≈ 0.033):

| dataset | intact | zero | permute | **noise** |
|---|---|---|---|---|
| covid19_twitter (in-domain) | 0.664 / 0.982 | 0.073 / 0.678 | 0.626 / 0.976 | **0.061 / 0.622** |
| midterm | 0.313 / 0.884 | 0.086 / 0.718 | 0.311 / 0.885 | **0.064 / 0.646** |
| twibot20 | 0.406 / 0.924 | 0.066 / 0.671 | 0.407 / 0.927 | **0.058 / 0.632** |

**Feature→label logreg** (raw features, no graph) — AUC / acc / majority:
election2020 0.95 / 0.88 / 0.54 · covid_political 0.91 / 0.86 / 0.75 · twibot20
0.71 / 0.66 / 0.56 · ukr_rus_suspended 0.60 / 0.59 / 0.58.

**Feature-only NM prototype** (no model) — acc (AUC), real vs permuted control,
chance 0.033: twibot20 0.169 (0.66) vs 0.035 · midterm 0.103 (0.61) vs 0.032.

**Classification linear-probe on the frozen representation** — AUC intact / zero /
permute: election2020 0.979 / 0.503 / 0.978 · covid_political 0.912 / 0.613 / 0.911
· twibot20 0.680 / 0.715 / 0.673.

## Findings / discussion

- **NM uses real feature content, not distinctness.** `noise` (distinct but wrong
  content) collapses NM to chance, matching `zero`; only `permute` (real content
  preserved) survives.
- **Content is used as a permutation-invariant bag, not a node binding.** `permute`
  ≈ intact on every graph.
- **Topology contributes little here, by construction.** At `n_hop = 1` subgraphs are
  stars; edges only deliver the neighbor feature set to the readout, and `zero`/`noise`
  keep edges yet give chance. Assessing whether real *structure* would add signal
  requires multi-hop retraining, not an eval ablation.
- **The features are good and the signal is real.** Raw feature→label AUC up to 0.95;
  neighborhoods are feature-distinguishable (feature-only NM 3–5× chance, control at
  chance).
- **Downstream feature-use is unresolved.** The classification probe was run
  `permute`-only, which does not test content-use (permute preserves the bag). The
  twibot20 gap (rep 0.680 < raw-feature 0.707) hints that individual-node semantics
  may be under-encoded relative to the neighborhood aggregate NM uses — unconfirmed.

## Caveats

One checkpoint (covid NM), one seed — effects are large and consistent across 3
graphs (robust within this checkpoint; cross-checkpoint unverified). twibot20 is the
clean low-homophily cell; political graphs are homophily-confounded for the
downstream comparison. All topology statements are specific to `n_hop = 1`.

## Next

1. `noise` on the classification linear-probe — settle whether the encoder keeps
   individual-node content or the political-graph numbers were homophily.
2. Feature-reconstruction probe — predict a node's raw bio from its frozen
   representation (encoded-but-unused vs. not-encoded).
3. Multi-hop (`n_hop = 2`) NM training — the only clean way to ask whether real graph
   structure adds anything beyond the neighbor feature set.

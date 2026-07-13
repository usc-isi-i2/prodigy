# feature_ablation — Findings

## Executive summary

We pretrain mostly on Neighbor Matching (NM) and asked whether NM (and the
downstream tasks) actually use the node features (GTE bio embeddings) or solve
everything from graph topology. Using a ladder of eval-time input ablations on a
fixed NM checkpoint, we find: **NM relies heavily on the real feature *content* of
a node's neighborhood — not on topology and not on mere node-distinctness.**
Destroying the real neighborhood content (`noise`) collapses NM to chance, exactly
like deleting features (`zero`), while scrambling only the feature↔node binding
(`permute`) is harmless. The bio features are informative (raw feature→label AUC
0.71–0.95), and neighborhoods carry genuine community-level feature signal.

**Correction:** an intermediate conclusion — "NM ignores feature content; the
encoder strips it" — was **wrong**. It rested on `permute`-invariance alone, and
`permute` preserves the content bag. The `noise` treatment overturned it. The
original motivation ("NM ignores features → add a feature-content objective") does
not hold: NM already uses feature content; if anything topology is the underused
channel, and at the deployed `n_hop = 1` there is barely any topology to use.

## Methodology

**Overall design.** We hold the checkpoint fixed and perturb only what the model
*sees* at eval. Episodes are still sampled from the real graph, so ground-truth
labels stay valid; the ablation changes the model input, not the task. In a
message-passing GNN "using features" is ambiguous — features can serve as node
*distinguishers*, as real *content* (used as a bag), or as *bound* content — so the
treatments vary features along two axes, **node distinctness** and **real
neighborhood content**, which lets them separate those hypotheses:

| condition | node distinct? | node↔feat binding | real neighborhood content |
|---|:--:|:--:|:--:|
| intact  | ✓ | correct | ✓ |
| permute | ✓ | scrambled | ✓ (preserved) |
| noise   | ✓ | n/a | ✗ (destroyed) |
| zero    | ✗ | — | ✗ |

**Treatments (hypothesis + motivation):**

- **`zero`** — replace every node feature with 0.
  *Hypothesis:* if NM survives, it runs on topology/structure alone; if it
  collapses, features are necessary. *Motivation:* crudest "are features needed at
  all" test — removes distinctness *and* content at once (a floor, not a clean
  isolation).

- **`permute`** — shuffle feature rows across nodes *within each subgraph*.
  *Hypothesis:* if NM drops, the model depends on the specific node↔feature
  binding; if invariant, it does not. *Motivation:* isolates the binding while
  preserving the subgraph's real feature multiset and distinctness. **Pitfall
  (learned the hard way):** permute preserves the content *bag*, so invariance does
  **not** rule out content use — which is exactly why `noise` is required.

- **`noise`** — resample every node's features from the *full graph's* feature
  distribution (`RandomNodeAttr`): distinct, in-distribution, but wrong content.
  *Hypothesis:* if NM holds → features are used only as distinguishers (H1); if NM
  collapses → the model uses the real neighborhood content (H2). *Motivation:* the
  decisive discriminator `permute` cannot provide — it destroys the real content
  while holding distinctness and feature scale fixed.

- **`edge-rewire`** (topology axis; wired as token `ER`, `AblateEdges("rewire")`) —
  randomize edges, keep features. *Hypothesis:* if NM drops, the model uses the
  real graph structure. *Motivation:* the mirror ablation on the topology axis.
  **Not run as a topology conclusion:** at `n_hop = 1` subgraphs are stars and the
  readout flows neighbors→center→supernode *through* the data edges, so perturbing
  edges is confounded with feature delivery and near-vacuous. A clean structure
  test needs multi-hop *retraining*, not an eval flag.

- **Feature→label probe** (`feature_label_probe.py`) — logistic regression from raw
  features to the node label, *no graph*. *Hypothesis:* AUC ≫ 0.5 ⇒ features carry
  real label signal. *Motivation:* separates "model ignores good features" from
  "features are broken/uninformative."

- **Feature-only NM probe** (`feature_only_nm_probe.py`) — prototype nearest-neighbor
  NM in raw feature space, *no model*, with a permuted-feature control.
  *Hypothesis:* ≫ chance ⇒ neighborhoods carry feature-discriminative content the
  model could exploit. *Motivation:* verifies the content the model relies on
  actually exists in the data, independent of the model.

**Reading rule.** `intact − ablated`: a large drop means the ablated property is
load-bearing. The `permute` vs `noise` contrast is the crux — both keep
distinctness, but only `noise` destroys real content, so `noise ≪ permute` ⇒
content is used; `noise ≈ permute` ⇒ distinguisher-only.

## Experiment setup

- **Checkpoint:** `nm_matrix_covid` (single-source covid NM,
  `state/nm_matrix_covid_28_06_2026_15_54_50/checkpoint/state_dict_50000.ckpt`).
- **Datasets:** NM on covid19_twitter (in-domain), midterm, twibot20;
  classification linear-probe on twibot20, covid_political, election2020;
  feature/feature-only probes on the labeled graphs.
- **Regime:** NM 30-way / 3-shot / test (chance 1/30 ≈ 0.033); classification
  linear-probe 20-shot. `n_hop = 1` (deployed default).
- **Harness:** `--ablate_features {none,zero,permute,noise}` (params) →
  `eval_ckpts_all_graph_tasks_tucker.py --ablate-features …`; ablated runs tagged
  `_ablZ/_ablP/_ablN`. Sweep via `run_feature_ablation_tucker.sh`, collate via
  `parse_feature_ablation.py`.
- **Env:** branch `experiment/feature-ablation`; run on Tucker (`prodigy` env) from
  the isolated worktree `/dataMeR1/phil/gfm/prodigy-featabl`.

## Results

Raw data: [`feature_ablation_results.csv`](feature_ablation_results.csv) (ablation
sweep), [`feature_label_probe_results.csv`](feature_label_probe_results.csv),
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
permute (**no `noise` yet**): election2020 0.979 / 0.503 / 0.978 · covid_political
0.912 / 0.613 / 0.911 · twibot20 0.680 / 0.715 / 0.673.

## Findings / discussion

- **NM uses real feature content, not distinctness.** `noise` (distinct but wrong
  content) collapses NM to chance, matching `zero`; only `permute` (real content
  preserved) survives. → the model needs the neighborhood's *actual* bios.
- **It uses content as a permutation-invariant bag, not a node binding.** `permute`
  ≈ intact on every graph.
- **Topology alone ≈ chance here — but that's architectural.** `zero`/`noise` keep
  edges yet give chance, and at `n_hop = 1` subgraphs are stars: edges only deliver
  the neighbor feature set to the readout. NM is a neighborhood-feature-content
  matching task by construction; "topology doesn't matter" is not a claim that the
  model ignores available structure.
- **Features are good, and the signal is real.** Raw feature→label AUC up to 0.95;
  neighborhoods are feature-distinguishable (feature-only NM 3–5× chance, control at
  chance). So NM's content-use is warranted, not a bug.
- **Methods takeaway:** never infer feature-disuse from `permute`-invariance alone —
  it preserves the content bag. Always run `noise`. This flipped our earlier
  conclusion, and it means the Phase-B "encoder strips content" claim (permute-only)
  is **not established** — the twibot20 rep 0.68 < raw 0.71 gap is only a hint that
  *individual-node* semantics may be under-encoded vs. the neighborhood aggregate NM
  uses.
- **Direction:** the "NM ignores features" premise is overturned. Open question is
  narrower — does the encoder keep individual-node semantics for low-homophily
  downstream? Resolve with (1) `noise` on the Phase-B probe, (2) a
  feature-reconstruction probe (raw bio ← frozen rep), (3) multi-hop retraining to
  ask whether real structure would ever add signal.

## Caveats

One checkpoint (covid NM), one seed — effects are large and consistent across 3
graphs (robust within this checkpoint; cross-checkpoint unverified). twibot20 is the
clean low-homophily cell; political graphs are homophily-confounded for the
downstream comparison. Phase B is permute-only. All topology statements are specific
to `n_hop = 1`.

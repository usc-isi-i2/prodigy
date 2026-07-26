# Research directions (2026-07-26)

What is worth *investigating*, organised by which knob it turns. Orthogonal to
[`directions_jul25.md`](directions_jul25.md), which sorts near-identical material by
*venue and claim*; this one is upstream of that sorting.

Tags: `[free]` analysis on artifacts we already have · `[eval]` eval-only sweep ·
`[train]` new pretraining.

---

## 1. Data — which graphs, and why does A→B transfer as it does?

*Prerequisite: why is any single number what it is? We cannot explain a difference between
two AUCs before we can explain one.*

- **1.1 Error anatomy** `[free]` — which episodes/nodes fail, per dataset. Hubs vs tail,
  missing/empty bios, label noise, ambiguous classes. Answers "why not 1.0".
- **1.2 Ceiling decomposition** `[eval]` — split the gap into task-intrinsic vs
  representation-intrinsic: supervised skyline, raw-feature floor, LLM-on-bios ceiling,
  untrained-encoder floor, all on the same splits.
- **1.3 Predict the pair** `[free]` — regress all 64 ordered pairs on data-side divergences
  (feature-cloud, language, degree, size, user overlap, missing-bio rate); decompose donor
  effect vs receiver effect vs interaction. Extends the ρ≈−0.92 pilot to the full matrix.
- **1.4 Intervene on the pair** `[train]` — graded on-manifold perturbation of the source's
  feature cloud (interpolation / rotation / subsampling) → dose-response curve in transfer.
  Converts 1.3 from correlation to cause; degree-preserving rewiring as the control arm.
- **1.5 Coverage** `[free]` — how much of a graph do 40k episodes actually touch, how
  hub-skewed is it, and does per-node error track visit frequency? Links the sampler to
  both the ceiling (1.1) and saturation (5.1).
- **1.6 Composition on the tasks we care about** `[eval]` — the ladder is established on
  the pretext metric; run the existing rungs on bot/suspension/leaning/regression. If the
  staircase is pretext-only, most of the program changes shape.
- **1.7 Selection payoff** `[train]` — the only item here that is a *deliverable* rather
  than a diagnostic: pre-register the top-k sources 1.3 predicts for a held-out target,
  train on just those, show it matches all-8 at a fraction of the compute. 1.1–1.6 explain
  transfer; this one is what a practitioner actually takes away, and no amount of
  explanation substitutes for it.
- **1.8 Predict transfer from the run, not the data** `[free]` — does held-out SSL
  performance (or its gradient norm, or activation statistics) rank downstream performance?
  1.3 predicts *ex ante* from data; this predicts *mid-flight* from the run, and is the
  cheaper signal if it holds. Caveat: for LP it is circular after the rescore (LP *is* an NM
  main effect) — restrict the claim to classification and regression.
- **1.9 Join, don't concatenate** `[train]` — every merge to date is a disjoint block-concat
  with no cross-source edges, so multi-graph pretraining has never actually been
  *multi-graph*: it is k graphs sharing a tensor. Link them on shared users (overlap is
  already computed as a divergence feature in 1.3) and re-run one ladder rung. If
  mixture≈max dissolves once the sources are genuinely connected, the composition rule is a
  statement about disjoint unions — worth knowing before it is the headline. Cheaper than
  the entity-resolution project it resembles: identity edges on exact user-id matches only,
  no fuzzy matching. Same underlying concern as §7.3, approached from the data side.

## 2. Objective — one objective set that clears chance 0-shot on every task

*Note the bar: "above chance" is too weak once floors exist (§6.1). The real target is
above the best cheap heuristic on every task simultaneously.*

- **2.1 Sum instead of rotate** `[train]` — weighted joint loss, weight sweep. Completes
  the 2×2 (rotation × summation) and is the direct test of whether dilution is a scheduling
  artifact.
- **2.2 Rotation granularity** `[train]` — >1 step per loss, per-episode vs per-k vs
  per-phase; unbalanced counts; curriculum (NM first, others as fine-tune).
- **2.3 Unify the prediction space** `[train]` — instead of predicting in different
  modalities, one matching/contrastive head over a shared target space, with topology and
  text as two *views* rather than two *losses*. The only item here that could remove the
  conflict rather than schedule around it.
- **2.4 Why mixing dilutes** `[free]`+`[train]` — per-objective gradient cosine at the
  shared encoder, CKA drift vs single-objective checkpoints. If the objectives genuinely
  conflict, no schedule fixes it — and that negative *is* the contribution.
- **2.5 Task-shaped pretexts** `[train]` — invent objectives for the tasks that never clear
  chance: profile-metric regression as a pretext (regression Spearman is ~0 today), an
  explicit pair-scoring head for LP instead of hoping NM emits one.
- **2.6 Objective × corpus** `[eval]` — is the best objective the same for every source
  graph? If the answer is no, "the best objective" is the wrong question and selection
  moves to the (graph, objective) pair.
- **2.7 Inject downstream tasks into pretraining** `[train]` — labelled episodes in the
  pretraining mix, so the model learns in-context task adaptation rather than only neighbour
  matching. Honest accounting: this abandons the zero-shot framing that makes every current
  result clean, and it is a separate paper (supervised multi-task pretraining). The fallback
  if 2.1–2.6 all fail to produce a generalist — not the first move.

## 3. Sampler — what the model is actually shown

*The single most attackable assumption in the program, and cheap to test.*

- **3.1 Receptive field** `[train]` — n_hop ≥ 2 with a large neighbour limit. Our
  "topology contributes little" conclusion is measured at n_hop=1, where it is nearly true
  by construction.
- **3.2 Structural inputs** `[train]` — degree / LapPE / RWSE as *features*, not as an
  objective. Separates "topology is unreadable" from "topology is uninformative".
- **3.3 Episode geometry** `[train]` — way/shot/query counts (30-way/3-shot is inherited,
  never tuned); does the pretext ceiling move with the task difficulty?
- **3.4 Sampling distribution** `[train]` — uniform sampling over a power-law graph leaves
  clusters and gaps; importance / degree-corrected sampling as the remedy, but only if 1.5
  shows the pathology. Prior evidence is discouraging (the cross-source probability sweep
  found plain within-source best).
- **3.5 Eligibility bias** `[free]` — episodes only draw centers with enough edges, so a
  power-law tail of low-degree nodes is *never trained on* and then evaluated on. Check
  whether per-node downstream error tracks training eligibility. Distinct from 3.4: that is
  the distribution over eligible nodes, this is who is eligible at all — and it is a bias we
  imposed, not one the graph has.

## 4. Representation — what is inside the embedding

- **4.1 Probe battery** `[free]` — on frozen embeddings: feature reconstruction, degree,
  community, source identity, label. What is preserved and what is discarded.
- **4.2 Source-identity probe** `[free]` — does merged pretraining make embeddings
  domain-separable? A candidate mechanism for the merged-vs-specialist tax.
- **4.3 CKA across the family** `[free]` — specialist vs merged vs mixed-objective
  checkpoints. Does adding a graph/objective *add* a subspace or *move* the existing one?

## 5. Scale — does any of this survive more of everything?

*Two payoffs, do not conflate them. Scale as **armor** ("big enough to deserve the words
foundation model") buys nothing at an ML venue and is a treadmill against industrial
backbones. Scale as **capacity control** ("mixture≈max is not an artifact of a 1M-param
encoder") is a claim. Same runs; report the second.*

- **5.1 Episodes** `[free]` — transfer saturation curve over existing checkpoints, read at
  the low end especially. If transfer is done by 1k episodes, most of the 40k budget is
  buying nothing and the training-steps question answers itself.
- **5.2 Parameters** `[train]` — 1M → 10–20M. Fit the *shape* of the curve, not just the
  endpoint: a few sizes early beats one big run, because the objection being answered
  ("your encoder is too small to hold two graphs") is about the trend, not the maximum.
- **5.3 Data volume** `[train]` — within-source subsampling 10/25/50/100%. Note the merged
  all-8 corpus is already tens of millions of nodes, so "train on 10M nodes" is satisfied on
  paper; the binding quantity is nodes *visited* (§1.5), not nodes available. Separates
  coverage from volume, which the multi-graph ladder confounds.

## 6. Evaluation — is the question even well-posed?

- **6.1 Floors and skylines everywhere** `[eval]` — untrained encoder, raw features, SGC,
  cheap LP heuristics, supervised skyline. The LP rescore already showed several arms sit
  *below* trivial baselines while reading as "above chance".
- **6.2 Seeds and MDE** `[train]` — nearly everything we quote is 1 seed; publish the
  minimum detectable effect so sub-noise effects get demoted on purpose, not by accident.
- **6.3 Frozen vs finetuned** `[eval]` — keep frozen as the headline (nothing is relearned
  at eval), add one finetuned arm on the pivotal comparison so the numbers are comparable
  to the supervised literature.
- **6.4 A task that requires topology** `[train]` — cascade participation / hashtag
  adoption. Without one, "0-shot on all tasks" is a claim about bios, and an LLM reader is
  the real opponent.
- **6.5 Audit the remaining evaluators** `[free]` — temporal LP carries the identical
  center-blind defect, latent. Anything else built on the metagraph/prototype path is
  suspect until checked.
- **6.6 A published GFM as external comparator** `[eval]` — one generic GFM
  (AnyGraph/OpenGraph/GCOPE-class) through our protocol. This is *comparability*, not
  invariance: it says whether our numbers are respectable, and tests none of our claims.
  Pick one, do not survey. Distinct from §7.2/7.3, which swap the model to stress the claim
  rather than to compete with it.

## 7. Generality — is this about our graphs, our model, or graphs?

- **7.1 Alien donor** `[train]` — a non-Twitter (OGB-class) graph in the merge. Tests
  whether the composition behaviour is a property of one platform.
- **7.2 Second backbone — layer swap** `[train]` — GraphSAGE → GIN/GAT inside the same
  episodic protocol. Cheapest invariance arm, worth one appendix row. Rules out "the rule
  is a property of the aggregator" — which was never the main suspect.
- **7.3 Second pretraining framework — no episodes** `[train]` — a GraphMAE-class
  reconstruction or DGI/GRACE-style contrastive objective over the same merged corpora.
  **The load-bearing invariance arm.** NM episodes never cross sources (merges are disjoint
  block-concats; the ladder ran at cross-source probability 0), so merged pretraining is a
  *union of per-source objectives with no coupling term* — under which mixture≈max is close
  to a property of the training construction rather than a discovered law. A framework with
  no source partition is the test that separates the two. Max survives → the claim is about
  multi-source pretraining; max dies → the claim scopes to episodic/retrieval SSL, which is
  narrower but still true, and better said by us than by a reviewer.
- **7.4 Second feature space** `[train]` — tweet-content embeddings instead of bio
  embeddings (pipeline already exists). Directly tests whether §1 is about *features* or
  about *bios*.

---

**Cheapest things that can most change the plan:** 1.6 (ladder on downstream tasks), 6.1
(floors), 5.1 (episode saturation), 2.4 (gradient conflict). All four are `[free]` or
`[eval]`, and each can falsify something we currently believe.

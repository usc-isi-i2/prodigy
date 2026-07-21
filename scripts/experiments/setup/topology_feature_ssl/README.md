# SSL that learns topology *and* features (not features only)

**Status: proposed — nothing run yet.** Benchmark-budget-aware plan. All decisions
are fixed below; the six arms run **flat — no staging, no early stopping** (one free
preview excepted, see Budget).

## Question

Can we find a self-supervised pretraining task whose frozen representations are
strong for **both** feature tasks (node classification, node regression) **and** a
topological task (static link prediction), by learning **both topology and
features** — instead of features only?

## Motivation (what we already know)

- **NM is feature-only by construction.** NM accuracy collapses to ~chance when
  node features are randomized/zeroed (`feature_ablation/`). NM's target is "which
  neighborhood-instance did this node come from," and neighborhood identity is
  carried by the *content* of the shared nodes (chiefly the anchor watermark).
  There is no topological solution to NM, so tuning the encoder can't fix it — the
  **objective** has to change (or the feature shortcut has to be corrupted; see B1).
- **The encoder can't represent counts/degree anyway.** The background GNN
  hardcodes `aggr="mean"` ([`models/get_model.py:41`](../../../models/get_model.py))
  and the readout is `global_mean_pool`
  ([`models/multilayer_gnn.py:56`](../../../models/multilayer_gnn.py)). Mean
  aggregation is degree-/count-blind, so counts, degree, and per-neighbor
  conjunctions are **not representable** regardless of the loss.
- **The graph is directed and the aggregation throws it away.** Retweet edges are
  directed (retweeter → retweeted; `n_retweets` weight). In-degree = **influence**,
  out-degree = activity. `preprocess` symmetrizes the sampling adjacency
  ([`experiments/sampler.py:15`](../../../experiments/sampler.py)) and NM walks
  `direction="inout"`, so in/out are conflated before the model uses them. Note
  direction is *not erased* — it survives in the `edge_attr` sign (+idx forward,
  −1−idx reverse) — so a directed-aware encoder can recover it from `edge_attr`
  without changing the sampler.

Three levers to fix this: **corrupt the feature shortcut** so NM is forced onto
topology (augmentation, cheapest — B1); make topology **representable** (encoder:
structural features, directed/count-aware aggregation — E1/E2); and make it **used**
(objective: generative/structural targets, not pure instance discrimination —
E3/E4).

## Fixed setup (identical across all arms)

- **Pretrain corpus — single, fixed:** the **3-way merged** retweet graph
  (ukr_rus + covid + midterm),
  `/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_retweet_graph.pt`. One
  pretrain per arm, **one seed**. Using one merged graph (not per-source
  pretraining) removes the pretrain-dataset multiplier. Caveat: the three in-domain
  eval sources are *in* this mix, so the cleanest **transfer** read is on the
  held-out datasets (twibot20, election2020); treat in-domain numbers as fit, not
  transfer.
- **Episode sampling — within-source, balanced** (NM arms B0/B1/E1/E2): every
  episode drawn from a single source with equal per-source share —
  `--neighbor_sampling_episode_source graph_id --neighbor_sampling_episode_source_weighting balanced`.
  Removes the cross-source shortcut and equalizes exposure. For the non-NM arms
  (E3/E4) sample subgraphs source-balanced for matched exposure.
- **Eval — the expensive unit:** freeze encoder, run the benchmark node
  **classification**, node **regression**, and **static link prediction** at a
  **single shot setting (10-shot; LP zero-shot + `--slp-n-query 4`, sparse-graph
  safe)**. One benchmark sweep per encoder. Metrics: classification → accuracy/AUC,
  regression → Spearman, static-LP → ROC-AUC. (LP is the direct topological *task*;
  regression on structure-linked targets is only a proxy, so both are reported.)
- **Free diagnostics — every frozen encoder, not benchmark jobs — and the PRIMARY
  evidence (see Acceptance):**
  - **2×2 ablation:** {real, random} features × {real, rewired} edges. "Learns
    both" ⇒ degrades under **both** halves. NM degrades only under random features
    — that's the signature to break.
  - **Capability probes:** planted single-rule synthetic graphs (count-threshold,
    existence, in-degree, out-degree, two-neighbor conjunction), linear-probed from
    the frozen rep.
- **Leakage control (mandatory for structural-input arms E1/E2).** E1 currently
  feeds degree-only `directed3` inputs (`in_deg`, `out_deg`, `log_deg`) because the
  full networkx feature set is not tractable on the 34M-node merged pretrain graph.
  Regression targets `followers` (≈ in-degree) and `statuses` (≈ out-degree) are
  near-copies of those inputs, so an encoder can win these **trivially by passthrough,
  not by learning.** So:
  - Add a **raw-structural-feature probe baseline**: linear-probe the raw
    `[in_deg, out_deg, log_deg]` vector directly onto
    each regression target, **no encoder**. E1/E2 count as "learned structure" only
    if the frozen rep **beats this baseline**.
  - Split the regression panel into **structure-linked** (`followers`, `statuses`)
    and **content-linked** (`account_age_days`) targets. Report E1/E2 gains on
    structure-linked targets **only relative to the raw-feature baseline**, and keep
    them out of E1/E2's headline "learned both" claim (the diagnostics + LP +
    content-linked regression carry that).
- **MFR masks has-bio nodes only (E3/E4).** ~20–23% of bios are zero-filled and the
  rate differs by source. Masked feature reconstruction over zero targets degenerates
  into "predict zero" and leaks per-source coverage, so E3/E4 draw mask targets
  **from has-bio nodes only**; missing-bio nodes are never reconstruction targets.
- **Held constant:** #hops, probe head, pretrain compute, shot count, eval
  datasets. One lever changes per arm (E2/E4 are declared composite; see Arms).

**Acceptance for "learns both" — diagnostics primary, benchmark confirmatory.** With
one seed and small expected benchmark deltas (the May aug delta was +0.008 AUC —
inside noise), the seed-robust qualitative signatures are the primary evidence and
the benchmark is directional/confirmatory:
1. **Primary (seed-robust):** the frozen rep **degrades under both halves** of the
   2×2 (not features-only like NM) **and** passes the structural capability probes.
2. **Confirmatory (directional):** improves the joint classification + regression +
   static-LP benchmark — especially **regression** (where NM is weak, because
   instance discrimination collapses continuous variation) and **static-LP** — and
   for E1/E2 clears the raw-feature leakage baseline. Treat a single-seed benchmark
   gain as suggestive, not decisive; if budget frees up, seed B0 and the winner ×3.

## Arms — six, three levers

**Augmentation lever** (control encoder + NM objective; cheapest — run first):

- **B0 — control.** mean-agg SAGE, bio features only, undirected message passing,
  no augmentation.
- **B1 — feature-shortcut corruption.** B0 + `NR0.3` (RandomNodeAttr: 30% of nodes
  get a random *real* feature vector). One knob (augmentation); encoder and objective
  unchanged. *Rationale:* corrupting features NM leans on forces it toward topology
  **without** the expensive objective change. **Use `NR`, not `NZ`** — `NZ0.3`
  (zeroing) aliases the already-zero missing bios, so the model treats it as
  "missing" and ignores it (the likely reason the May `NZ0.3` run was ~null); `NR`
  is a plausible-but-wrong feature the model can't detect by a zero-check. *Hyp:* NM
  stops being ~chance under **random-feature** 2×2 half; regression ↑. **If B1 moves
  the joint benchmark, cheap aug partially substitutes for E3/E4; if it stays flat,
  that is strong evidence the objective *must* change — which de-risks the expensive
  arms either way.**

**Encoder axis** (objective = NM throughout; isolates *representable*):

- **E1 — directed structural input features.** Add per-node in-deg, out-deg,
  and log-deg (`directed3`); else B0. Subject to the leakage control above. *Hyp:*
  NM stops being ~chance under **random bio features**; regression ↑
  on structure-linked targets **beyond the raw-feature baseline**; classification
  ~flat.
- **E2 — expressive directed aggregator (composite arm).** E1 + a *package* of three
  coupled changes: `aggr` mean→sum→**PNA**, in/out neighbors aggregated separately,
  readout mean→mean⊕sum⊕max. Not a single knob — if E2 helps, PNA vs directed-split
  vs multi-readout is **not** individually attributed here; deferred to A1/A2/A4.
  *Hyp:* count/existence capability probes jump and the **topology half of the 2×2
  widens**; the in/out split helps influence-linked targets specifically.

**Objective axis** (encoder = **E2's**, fixed; isolates *used*):

- **E3 — masked feature reconstruction (GraphMAE-style).** Swap NM → MFR (mask incl.
  center, scaled-cosine, **has-bio targets only**). *Hyp:* beats NM by a **larger
  margin on regression than classification**, and — unlike NM — degrades under
  **both** 2×2 halves.
- **E4 — multi-task: MFR ⊕ directed link-prediction ⊕ structural-property
  prediction (composite arm).** Structural head predicts a masked node's
  in-deg/PageRank/influence; the LP head must read the **original directed edges /
  `edge_attr` sign, not the symmetrized sampling adjacency** (else "directed LP"
  isn't directed). Three objectives at once — component attribution deferred to
  A1/A2. *Hyp:* best **average** over cls+reg+LP, and the only arm that degrades
  under both 2×2 halves *and* passes all capability probes.

Reading the chain: B0→B1 attributes the cheap augmentation lever (control encoder +
objective); B0→E1→E2 attributes the encoder pieces (all under NM); E2→E3→E4
attributes the objective (all under the capable encoder). E2 and E4 are composite
packages — within-package attribution is deferred to the A-arms.

## Budget

- **Free preview first (uses an existing checkpoint, no new train):** the
  pretrain-strategy benchmark already has `task_transfer_covid_nm` and
  `task_transfer_covid_fp` (masked feature prediction ≈ E3's objective).
  Compare **nm vs fp on regression** on the existing frozen encoders before
  committing pretrains — if MFR already beats NM there, E3's core hypothesis is
  pre-validated for ~zero cost. Doesn't violate the flat-run design; it just reads a
  prior we already own.
- **Train:** **6 pretrains** (B0, B1, E1, E2, E3, E4) — 1 corpus, 1 seed.
- **Eval:** 1 benchmark sweep per encoder at 10-shot = **6 sweeps**, each fanning
  out into eval jobs ≈ classification(`D_cls`) + regression(`D_reg` × targets) +
  static-LP(`D_lp`):
  - focused 5 datasets (3 in-domain: ukr_rus, covid, midterm + held-out twibot20 +
    election2020): ~40 jobs/encoder → **~240** total.
  - all applicable datasets: ~68 jobs/encoder → **~410** total.
  - the multi-target regression panel is the cost driver; trimming to 3
    representative targets (followers=influence, account_age_days=age,
    statuses=activity) halves the regression share. LP adds ~1 job/applicable
    dataset.
- **Diagnostics:** free (offline on the frozen encoder).

## Expected results format

Everything is keyed by **arm** (`B0, B1, E1, E2, E3, E4`) so the reading-chain
deltas are direct table subtractions. Benchmark CSVs land under
`scripts/experiments/analysis/{node_regression,static_link_prediction,node_classification}/data/`
(one `model=<arm>` column, as in the pretrain-strategy benchmark); diagnostics land
in `scripts/experiments/analysis/topology_feature_ssl/` and are joined in that folder's
notebook. Three tables:

**T1 — Benchmark (confirmatory).** Rows = arms; columns split feature vs topological
and structure-linked vs not, all over the focused 5 datasets with held-out
(twibot20/election2020) broken out from in-domain:

| arm | cls (acc/AUC) | reg-content: age (Spearman) | reg-structure: followers/statuses (Spearman **Δ vs raw-feature baseline**) | static-LP (ROC-AUC) |
|---|---|---|---|---|

Structure-linked regression is reported **only as Δ over the raw-feature probe
baseline** (the leakage control) — a raw number there is uninterpretable for E1/E2.

**T2 — 2×2 ablation (primary, seed-robust).** Rows = arms; cells = **fraction of the
real/real benchmark retained** under each corruption:

| arm | real feat · real edge (=1.00 ref) | **random feat** | **rewired edge** |
|---|---|---|---|

The signature to break: B0/NM stays high under *rewired edge* and collapses under
*random feat* (features-only). "Learns both" ⇒ **drops materially under both**.

**T3 — capability probes (primary).** Rows = arms; cells = linear-probe score on
each planted single-rule graph:

| arm | count-threshold | existence | in-degree | out-degree | 2-neighbor conjunction |
|---|---|---|---|---|---|

## What we can conclude — the reading

Each comparison licenses exactly one claim; read them as a chain, primary
(T2/T3) over confirmatory (T1):

- **B1 − B0 (aug lever).** If B1's *random-feat* 2×2 cell rises and reg-structure/LP
  improve → **the feature shortcut, not the objective, was the binding constraint**;
  cheap data-side corruption recovers topology. If B1 is flat on all three → the
  shortcut is not removable by augmentation, i.e. **the objective must change**
  (positive support for E3/E4, not a null result).
- **E1 − B0, against the baseline (structure representable, injected).** Reg-structure
  Δ **> 0 over the raw-feature baseline** → the encoder *learned* to use injected
  structure, not passthrough. Δ ≈ 0 → leakage/passthrough only; E1 "worked" for the
  wrong reason and the claim is withheld.
- **E2 − E1 (structure representable, counted).** T3 count/existence/in-out cells
  jump and the T2 *rewired-edge* drop widens → count-capable, directed aggregation
  adds representational capacity a mean encoder cannot. (Composite arm: attributes to
  the *package*, not PNA-vs-split-vs-readout individually.)
- **E3 − E2 (objective used).** NM→MFR degrades under **both** 2×2 halves (not
  features-only) and beats NM by a larger margin on **regression than
  classification** → a generative objective makes topology *used*, not just
  representable.
- **E4 − E3 (topological task specifically).** Static-LP and structural-probe cells
  rise with ≤ noise loss on feature tasks → the multi-task structural/LP heads buy
  the topological task without sacrificing the feature tasks. This is the arm that
  must clear the **joint** bar: feature tasks up **and** LP up **and** both 2×2
  halves down **and** probes pass.

**The joint criterion is the point.** An arm that lifts LP by dropping regression, or
passes probes while losing classification, has **failed** the cross-task goal — which
is why T1 is reported as a per-task vector, never a single mean, and the headline is
`min(feature_score, topological_score)`, not their average.

## Takeaway — the possible headline stories

The design is built so that *whichever* pattern appears is a publishable, decision-
shaped conclusion, not just a win/lose:

- **"Objective is the binding constraint"** — E3/E4 clear the joint bar, B1/E1/E2 do
  not. Cross-task GFM transfer needs a generative/multi-task objective; encoder
  tuning and augmentation are insufficient. (This is the prior we expect.)
- **"Cheap augmentation suffices"** — B1 matches E3/E4 on the joint bar. The
  expensive objective change is unnecessary; corrupting the feature shortcut is
  enough. Cheapest possible positive result — hence B1 runs first.
- **"Capacity, not objective, was the bottleneck"** — E2 clears the bar under plain
  NM. The problem was a count-blind encoder all along; the objective was fine.
- **"Features dominate; topology unrecoverable at this budget"** — no arm degrades
  under both 2×2 halves and LP stays weak everywhere. A bounded negative result that
  says social-graph SSL at this scale/depth is feature-limited, and points at the
  deferred levers (depth A3, temporal A5, positional encodings) as the next place to
  look.

The single sentence this experiment is designed to produce: *"On merged social-media
retweet graphs, learning representations that transfer to **both** feature and
topological tasks requires **[the lever that cleared the joint bar]** — and we can
show it, because that arm is the only one that degrades under both halves of the 2×2
while the others remain features-only."*

---

## Appendix

### Deferred arms (not in the main six)
- **A1 — structural-property prediction as a standalone auxiliary** (attribution for
  E4's structural head).
- **A2 — directed link-prediction alone** (E4 component; expected weaker pure-feature
  regression than MFR).
- **A3 — depth 1 → 2** (recursive/global quantities: influence propagation,
  community; watch oversmoothing).
- **A4 — redundancy-reduction contrastive** (CCA-SSG / VICReg-graph) as a
  regression-safe alternative to NM/InfoNCE.
- **A5 — temporal structure** (edge timestamps / recency), only if a downstream
  label is time-dependent.
- **A6 — structural augmentation (`ND0.3`, DropNode) alone** — B1 tests feature-side
  corruption; this tests structure-side. Fold in only if B1 shows augmentation is a
  live lever.
- **A7 — augmentation stacked on the winning E-arm** — the combined `NR`/`ND` +
  best encoder/objective, to check additivity once the single-lever arms are read.

### Capability taxonomy this plan targets

| Primitive | Plain meaning | Supplied by |
|---|---|---|
| Preserve (no collapse) | keep self + neighbor content + structure jointly decodable | E3/E4 (generative) |
| Per-neighbor predicate | learned threshold on each neighbor | any msg-MLP encoder |
| Fraction / count / existence | mean / sum / max over a predicate | **E2** (mean→sum→PNA + multi-readout) |
| Cross-neighbor binding | which neighbor has which property | attention encoder (partial; A3) |
| Recursion / depth | iterate; fixed points (influence, centrality) | depth (A3) |
| Structural position | degree/clustering/centrality/role | **E1** (inject) + E2 (count) |
| Structure×content join | gate a feature on a structural fact | E1 + E3 (kept jointly) |
| Direction | in- vs out-degree (influence vs activity) | **E1/E2** (directed) |
| Reduce feature-shortcut reliance | force topology use by corrupting content | **B1** (`NR` aug) |

Not covered by any arm: edge **type/weight** beyond direction, **temporal**
dynamics (A5), **global/positional** encodings. Add only if the benchmark demands.

### Why NM cannot be salvaged by the encoder alone
NM's positives are same-neighborhood nodes whose X^D correlate through shared
ego-graph content — mostly the anchor watermark, a *content* signal by
construction. Strengthening the encoder can't make NM's target depend on topology;
hence E3/E4 change the objective. B1 attacks the same problem from the data side —
corrupting the content shortcut rather than changing the loss — and is the cheap
test of whether the objective change is even necessary.

# multitask_ssl_pairs — findings

## 1. Executive summary

The 3-way SSL rotation (**MIX**) was the only pretraining that produced a
*generalist* encoder — near-best node classification **and** the only arm with
real zero-shot link-prediction (topological transfer none of its constituent
objectives achieve alone). This experiment asks **whether that emergent
link-prediction requires all three objectives, or whether some pair already
unlocks it** — and if a pair does, which objective is the driver.

We trained the three pairwise rotations (NM+CL, NM+FP, CL+FP) at matched compute
and scored the complete 7-arm subset lattice of {nm, cl, fp} — 3 singles, 3 pairs,
1 triple — in a single frozen-encoder eval sweep.

**Result: the emergent link-prediction is a genuine 3-way synergy.** No single and
**no pair** clears chance on static link-prediction (AUC 0.23–0.47, chance 0.50);
only the full rotation does (0.76, on all four eval datasets). Capability is
**non-monotonic** in the number of objectives — on the joint feature+topology bar
the pairs are, if anything, *worse* than the singles (singles 0.42 → pairs 0.32 →
triple 0.76). Classification remains an **NM** property that is preserved under
combination; regression a weak **FP** property. The feature+topology generalist is
an emergent, super-additive property of the *complete* objective set, not of any
proper subset.

## 2. Methodology

### 2.1 Experiment setup

- **Corpus / encoder.** One encoder per arm on the merged 3-way retweet corpus
  (ukr_rus + covid + midterm), bio-only 768-d GTE node embeddings, mean-aggregation
  SAGE (1 layer, 1 hop), global episode sampling (no within-source confinement).
  Identical across all arms — arms differ *only* in the SSL objective(s).
- **SSL primitives** (one objective = one episode):
  - **nm** — `NeighborTask`, no augmentation, metric / instance-discrimination loss
    (predict a node's neighbors).
  - **cl** — `ContrastiveTask`, two feature-augmented views (NZ0.2), metric loss.
  - **fp** — `ContrastiveTask` episode with whole-node feature masking (NZ0.3),
    reconstruction loss via an auxiliary head.
- **Matched compute.** Every arm gets **40k episodes total** (`batch_size=1`, one
  objective per gradient step). Singles spend all 40k on one objective; pairs split
  ~20k/20k in a 1:1 per-episode rotation; MIX splits ~13.3k×3 at 1:1:1. So the only
  thing that varies across the lattice is *how many distinct objectives share a
  fixed budget* — any capability change is attributable to objective composition,
  not compute. Checkpoints at 10k/20k/30k; all arms evaluated at the **30k** ckpt.
- **Config-only comparability.** Each pair reuses the *exact* `nm_fp_cl` rotation
  machinery with one objective's count zeroed (`MultiTaskSplitBatch` never samples a
  0-count task; the fp reconstruction loss only fires on episodes the collator tags
  `mix_is_fp`). A pair is therefore a byte-identical code path to the NM/CL/FP/MIX
  arms — no core-code changes — so all 7 arms are directly comparable.
- **Evaluation.** Freeze each encoder and run the joint benchmark over the
  focused-5 datasets (3 in-domain: ukr_rus, covid, midterm; 2 held-out: twibot20,
  election2020):
  - **node classification** — 10-shot, ROC-AUC → *feature* axis
  - **static link-prediction** — 0-shot, n_query 4, ROC-AUC → *topological* axis (the headline)
  - **node regression** — 10-shot, log1p, Spearman ρ → secondary, noisy *feature* axis
  All 7 arms (pairs here + NM/CL/FP/MIX checkpoints reused from
  `multitask_ssl_rotation`) were scored in one sweep under identical conditions.
  Single seed (seed 0), matching the rotation run.

### 2.2 Treatments — definitions and hypotheses

Each arm is a per-episode round-robin over its objective set at matched 40k
compute. The **pairs are the treatments**; the singles are lower controls and MIX
is the upper anchor.

| arm | group | definition | hypothesis |
|---|---|---|---|
| **NM** | control (k=1) | neighbor matching only, 40k | classification specialist; chance LP; ~0 regression (from the rotation run) |
| **CL** | control (k=1) | contrastive only (NZ0.2), 40k | weak on every axis — NZ0.2 is a near-trivial pretext |
| **FP** | control (k=1) | masked feature prediction only (NZ0.3), 40k | regression specialist; chance classification and LP |
| **NM+CL** | **treatment (k=2)** | nm ⊕ cl, 1:1 (two metric objectives, no reconstruction) | keeps NM's classification; tests whether a *second metric view* alone adds topological transfer |
| **NM+FP** | **treatment (k=2)** | nm ⊕ fp, 1:1 (metric + reconstruction) | the two individually-useful specialists combined — **the most likely pair to unlock LP** if two objectives suffice; expected best pair, plausibly a generalist |
| **CL+FP** | **treatment (k=2)** | cl ⊕ fp, 1:1 (contrastive + reconstruction, **no nm**) | isolates whether nm is *necessary* — if LP emerges here, neighbor-matching is not required; if it collapses, nm is implicated |
| **MIX** | anchor (k=3) | nm ⊕ cl ⊕ fp, 1:1:1, 40k | known generalist with emergent LP (0.76) — the target the pairs are measured against |

**Overarching hypotheses** (pre-registered reasoning):

- **H1 — interpolation.** Capability scales with the number of objectives; pairs
  land *between* singles and MIX (partial credit toward the generalist).
- **H2 — pairwise sufficiency.** If emergent LP is a pairwise interaction, at least
  one pair clears static-LP; *which* pair names the driver (e.g. only NM+FP →
  nm×fp interaction; only fp-containing pairs → fp drives it).
- **H3 — 3-way necessity** (the alternative to H1/H2). If no pair clears LP and only
  MIX does, all three objectives are *jointly* necessary — a 3-way synergy.

## 3. Results

**T1 — frozen-encoder transfer over the {nm, cl, fp} lattice** (mean over eval
datasets, test split; cls = 2 labeled datasets, reg = 4 datasets × 3 targets,
sLP = 4 datasets). Reproduce: `python aggregate_results.py --plotting-root results`
(full dump in `results/lattice_table.txt`).

| arm | k | cls AUC | reg ρ | **sLP AUC** | min(cls,sLP) | group |
|---|---|---|---|---|---|---|
| NM   | 1 | **0.810** | −0.001 | 0.467 | 0.467 | single |
| CL   | 1 | 0.638 | −0.128 | 0.332 | 0.332 | single |
| FP   | 1 | 0.492 | **0.166** | 0.449 | 0.449 | single |
| NMCL | 2 | 0.800 | −0.144 | 0.305 | 0.305 | pair |
| NMFP | 2 | 0.802 | −0.098 | 0.424 | 0.424 | pair |
| CLFP | 2 | 0.601 | 0.110 | 0.227 | 0.227 | pair |
| **MIX** | 3 | 0.795 | 0.097 | **0.759** | **0.759** | triple |

**Static-LP ROC-AUC per dataset** (0-shot, chance 0.50) — MIX is the only arm above
chance, and it clears it *everywhere*:

| dataset | NM | CL | FP | NMCL | NMFP | CLFP | **MIX** |
|---|---|---|---|---|---|---|---|
| covid19_twitter | 0.406 | 0.366 | 0.449 | 0.320 | 0.395 | 0.228 | **0.755** |
| midterm | 0.487 | 0.417 | 0.433 | 0.376 | 0.445 | 0.310 | **0.676** |
| twibot20 | 0.491 | 0.259 | 0.381 | 0.315 | 0.471 | 0.202 | **0.745** |
| ukr_rus_twitter | 0.484 | 0.288 | 0.534 | 0.210 | 0.387 | 0.168 | **0.861** |

**Capability vs. number of objectives** (mean over arms at each k):

| k | arms | cls AUC | reg ρ | sLP AUC | **min bar** |
|---|---|---|---|---|---|
| 1 | NM, CL, FP | 0.647 | 0.012 | 0.416 | **0.416** |
| 2 | NMCL, NMFP, CLFP | 0.734 | −0.044 | 0.319 | **0.319** |
| 3 | MIX | 0.795 | 0.097 | 0.759 | **0.759** |

**Joint generalist bar** `min(feature = cls AUC, topological = sLP AUC)`, ranked
(chance 0.50 both axes; > 0.6 = genuine dual-capability generalist):

- MIX **0.759** ⟵ only generalist · NM 0.467 · FP 0.449 · NMFP 0.424 · CL 0.332 ·
  NMCL 0.305 · CLFP 0.227 — every arm except MIX is bottlenecked by sLP at chance.
- best single = NM (0.467) · best pair = NMFP (0.424) · MIX (0.759).
  **best-pair − best-single = −0.042; MIX − best-pair = +0.335.**

## 4. Findings / discussion

**H1 and H2 are rejected; H3 is supported — emergent LP is a 3-way synergy.**

1. **No pair reproduces the emergent link-prediction.** Every single and every pair
   sits at or below chance on static-LP (0.23–0.47); only MIX clears it (0.76, on
   all 4 datasets). The pair we expected to win (NM+FP, 0.424) does not, and the
   nm-free pair (CL+FP) is the *worst* arm in the whole lattice (0.227) — so no
   two-objective interaction accounts for the transfer.

2. **Pairs are not partial credit — capability is non-monotonic.** Adding a second
   objective does not push sLP toward MIX; several pairs are *worse* than their own
   constituents (NMCL 0.305 < NM 0.467; CLFP 0.227 < both CL and FP). On the joint
   generalist bar the trend is singles 0.42 → pairs **0.32** → triple 0.76: two
   objectives *dilute*, three *synergize*. The −0.04 best-pair-vs-best-single gap
   versus the +0.34 leap to MIX is the whole story in two numbers.

3. **Per-axis attribution.** Classification is an **NM** property and is preserved
   under combination — NM 0.810 → NMCL 0.800, NMFP 0.802, MIX 0.795 (all ≈ NM),
   while FP alone collapses to chance (0.492); adding cl/fp costs classification
   almost nothing. Regression is a weak **FP** property (FP 0.166, the only clearly
   positive single) but noisy for everyone (|ρ| ≤ 0.17, several negative). The
   topological axis (static-LP) is the *only* one that is MIX-exclusive.

4. **Interpretation.** The three objectives are individually specialists, and
   pairwise still specialists or worse; the topological transfer surfaces only when
   all three rotate together. The most plausible mechanism is that the complete
   3-way rotation acts as a regularizer that forces a representation no single loss
   — and no pair of losses — has any pressure to build. This is the strongest form
   of the "learns both feature and topology" thesis: it is not that more objectives
   monotonically help, but that the *complete* set unlocks a capability none of its
   subsets touch.

### Caveats

- **Single seed** (seed 0), matching the rotation run. The sLP gap is large and
  consistent across all 4 datasets (MIX ≥ 0.68 everywhere; every subset ≤ 0.53), so
  the qualitative result is not seed noise; exact values are single-draw.
- The `results/lattice_table.txt` "marginal sLP by objective" read (nm +0.153, fp
  +0.097, cl −0.041) is **confounded** — MIX is the only high-sLP arm and contains
  all three objectives, inflating every "with" average. It is *not* a clean
  per-objective additive effect; the real story is the 3-way interaction.
- Regression is weak/noisy for all arms — treat reg deltas as within-noise.

### Follow-ups this teed up

- **Edge/feature ablation on MIX** (no retraining): re-run static-LP with edges
  rewired (expect LP → chance) vs. features permuted (expect LP survives) to *prove*
  the emergent transfer is topological rather than a feature artifact.
- **Multi-seed** hardening of the k=1/2/3 trend (currently 1 seed).

### Provenance

Pairs trained 2026-07-14 in worktree `/dataMeR1/phil/gfm/prodigy-mtp` (branch
`mtp-run`), 40k episodes each, evaluated at the 30k ckpt. NM/CL/FP/MIX checkpoints
reused from the `multitask_ssl_rotation` run (`prodigy-mtr/state/mtr_*`); all 7 arms
scored in one eval sweep. `results/` holds the three benchmark CSVs (keyed by
`model` = arm, force-added past the `*.csv` ignore) and the aggregator dump; the
tables above regenerate from them via `aggregate_results.py --plotting-root results`.

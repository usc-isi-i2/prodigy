# Structured directions → contributions (2026-07-25)

Companion to [`./related_work/RELATED_WORK_AND_GAPS.md`](./related_work/RELATED_WORK_AND_GAPS.md)
(claims C1–C6, experiments E1–E14) and
[`./state_doc_jul22.md`](./state_doc_jul22.md).

This doc does two things: (1) split the program into **two papers with disjoint
contributions** — a CSS/social paper (ICWSM) and an ML paper (NeurIPS/ICML/ICLR) — and
(2) assign every raw direction to one of them, restated as a *hypothesis with a
falsifiable prediction and a kill criterion*, because a direction that cannot be killed
cannot be a contribution.

Venue tags used throughout: **[I]** = ICWSM paper, **[M]** = ML paper, **[I+M]** = shared
asset (run once, framed differently in each), **[—]** = neither.

---

## 0. The framing problem (read this first)

The venues do not accept "we built a graph foundation model for social networks."
A system + a leaderboard is a WWW/ICWSM/KDD-applied paper. At NeurIPS/ICML/ICLR the
accepted shapes are:

1. **A law** — a quantitative regularity that holds across conditions and predicts
   something (scaling laws, mixing laws, emergence curves).
2. **A mechanism** — an interventional explanation of *why* a known phenomenon happens,
   where the intervention is the contribution, not the observation.
3. **A negative result with a fix** — "the field believes X; under a corrected protocol
   X is false, and here is the corrected protocol."

Our existing results already sit on all three (composition rule / feature-space
mechanism / evaluator forensics). **Most items in the direction list are engineering
knobs, which fit none of the three.** The sorting below is largely about separating
those two populations.

### The three claim families we hold

| Family | One-line claim | Assets we already hold | Missing |
|---|---|---|---|
| **A. Coverage, not accumulation** | Multi-graph SSL composes as `≈ max over sources`, not as a sum; a source's marginal value is predictable and usually zero. | 3-order ladder, 8×8 donor/receiver matrix, entry-jump stats (p≈5e-7), divergence predictor (ρ≈−0.92) | volume-vs-coverage control, functional-form fit, saturation-in-scale, downstream-task ladder |
| **B. Relational signal lives in feature space** | For SSL transfer on text-attributed graphs the graph is a feature-gathering device; topology contributes near-nothing, and forcing its use fails. | feature ablations (zero/noise→chance, shuffle≈no-op), divergence axes, failed topology-forcing arms | nhop>1 rerun, untrained/SGC floors, dose-response intervention, a topology-*requiring* task |
| **C. The evaluator was broken** | Episodic label-prototype LP evaluation is center-blind and degree-confounded; repairing it inverts a published-style "emergent synergy" into an NM main effect. | full rescore, valid-only lattice, defect writeup | HeaRT negatives, upstream PRODIGY audit, corpus release |

**"A GFM for social networks" is the trap.** It converts every direction below into a
required experiment (scale, finetuning, baselines, more tasks) and still competes against
industrial backbones on their turf. It is the *intro framing and venue vehicle for the
ICWSM paper*, never a contribution in either paper.

### The RQs, as hypotheses

- **RQ1 (composition).** When k source graphs are pretrained jointly, is target
  performance additive in sources, or `max` over sources?
  *Prediction:* per-target performance jumps once and only when that target's best donor
  enters the mix; post-entry level is invariant to order and to k. *Kill:* a monotone
  gain with k after best-donor entry, or order-dependence post-entry.
- **RQ2 (predictability).** Can the marginal value of a source be predicted *before*
  pretraining, from a divergence measured on data alone?
  *Prediction:* feature-space divergence ranks donors (ρ≈−0.9) and beats structural
  predictors (EGI/W2PGNN/TMD/degree-KS); predicted-best k=2–3 subset ≈ all-8. *Kill:* a
  structural predictor matches or beats it, or selection payoff is flat.
- **RQ3 (mechanism).** Is transfer carried by feature content or by topology?
  *Prediction:* graded feature-space interventions produce a monotone dose-response in
  transfer; degree-preserving rewiring at matched degree sequence does not. *Kill:*
  rewiring hurts as much as feature corruption (then structure matters and B dies).
- **RQ4 (objectives, secondary).** Does combining SSL objectives help?
  *Prediction:* combination dilutes the task-aligned objective; neither summing gradients
  nor scheduling rescues it — i.e. the graphs-vs-objectives asymmetry is real. *Kill:*
  joint-loss weighting recovers single-objective performance on the aligned task.

---

## 1. The two-paper split

The three claim families do not fit one paper, and they do not fit one *audience*. The
split below is drawn so that neither paper's contribution is a subset of the other's —
which matters because ICWSM is archival and would otherwise self-scoop the ML paper.

**The dividing line: the ICWSM paper reports *which* and *how much*; the ML paper reports
*why* and *whether it is a law*.** Observations, corpus, task suite, and the practitioner
recipe go left. Functional form, capacity invariance, gradient geometry, and
interventional causality go right.

### Paper I — ICWSM 2027: *"Which event graph should you pretrain on?"*

- **Audience question:** a CSS researcher has a new event corpus (an election, a crisis, a
  protest) and limited labels. Which existing graphs should they pretrain on, and what do
  they get for free?
- **Contributions:**
  1. **Corpus + task suite.** 8 event-scale social graphs; bot / suspension / political
     leaning / regression / LP under one frozen protocol, with supervised skylines printed
     alongside.
  2. **Transfer structure across events** — the 8×8 donor/receiver matrix. *Covid and
     Ukraine are near-universal donors; HK is isolated; every graph has a strong
     specialist; transfer is asymmetric.* Reported as **empirical structure of this
     domain**, not as a general law.
  3. **A source-selection recipe that works** — feature divergence ranks donors ex ante;
     the predicted-best 2–3 sources ≈ all 8 (E8). This is the paper's payoff.
  4. **An evaluator warning + fix** (family C) — episodic label-prototype LP evaluation is
     broken, here is the repair, here is the inverted conclusion. CSS venues reward
     protocol hygiene, and this lands better here than as an ML appendix.
- **Explicitly NOT in Paper I:** the word "law", the functional-form fit, capacity
  control, gradient geometry, dose-response interventions, the alien-donor test. Paper I
  may *show* the staircase as descriptive evidence for the recipe; it must not *claim* the
  composition rule.
- **Deadline:** **Sep 15, 2026** (verified), second round Jan 15, 2027. ~7 weeks out.
- **Feasibility:** achievable with existing checkpoints plus eval-only work — *conditional
  on T1.6 (downstream ladder) and P0 seeds/floors landing in the next 3 weeks.* If T1.6
  slips, take the Jan 15 round rather than submitting on 1 seed.

### Paper M — ICML 2027 / NeurIPS 2027: *"Coverage, not accumulation"*

- **Audience question:** how does multi-source self-supervised pretraining compose, and
  what is actually being transferred?
- **Contributions:**
  1. **A composition rule with a functional form** — mixture ≈ max over components, fitted
     against additive and smooth-in-proportions (LM mixing-law) alternatives with model
     selection (E5); entry-jump statistics; order-invariance; volume-vs-coverage control
     (E4).
  2. **Invariance of the rule** — to capacity (10×, E14/T1.1b), to architecture (GIN row),
     to an out-of-family donor (E11). This is what upgrades "our 8 Twitter graphs behave
     this way" into a claim.
  3. **Mechanism** (family B) — feature-space dose-response vs degree-preserving rewiring
     (E7), with the nhop≥2 and structural-featurization controls (E9) and untrained/SGC
     floors.
  4. **The graphs-vs-objectives asymmetry, explained** — gradient geometry + CKA (T1.2),
     plus the 2×2 completion (joint loss vs rotation vs singles, E10) as counter-evidence
     to ControlG.
- **Deadlines (verify before planning around them):** ICML 2027 ≈ **Jan 2027**;
  NeurIPS 2027 ≈ **May 2027**. ICLR 2027 (≈ late Sep 2026) **collides with ICWSM and
  should be skipped** — four months of separation is what makes the two-paper plan
  workable. TMLR is the no-deadline fallback.

### Shared assets, and how each paper frames them

| Asset | Paper I framing | Paper M framing |
|---|---|---|
| 8×8 donor/receiver matrix | "transfer structure between events" (result) | input to the predictor comparison (E6) and evidence for the rule |
| Ladder + entry jumps | descriptive: "adding graphs stops helping" | the composition rule, fitted and stress-tested |
| Feature divergence predictor | selection recipe for practitioners | contrarian result vs the structure-first GDA canon, with interventions |
| Evaluator forensics (C) | full section + released fix | one appendix paragraph + citation to Paper I |
| Frozen/in-context protocol | label-efficiency argument for CSS | clean-transfer argument (nothing relearned at eval) |

### The self-scoop discipline (non-negotiable)

Paper M's contribution list must survive the sentence *"the authors already published the
donor matrix and staircase at ICWSM."* It does, if and only if Paper M's claims are the
functional form, the invariances, the mechanism, and the gradient geometry — none of which
appear in Paper I. Concretely: **Paper I gets no `max`-vs-additive model comparison, no
capacity arm, and no intervention.** If that feels like Paper I is being starved, the
right response is to strengthen its corpus/task/recipe axis, not to borrow from M.

**arXiv:** post Paper I's preprint at ICWSM submission. Per the scoop watchlist the
mechanism (B) and objective-dilution windows are closing, so if Paper M slips past ICML,
post its mechanism section as a standalone preprint rather than holding it for NeurIPS.

---

## 2. Tier 1 — directions that can carry a claim

Each of these makes a *specific* claim sharper or closes the one confound a reviewer
will name. Nothing else in the list does. Venue tag on each heading.

### T1.1 **[M]** Saturation in every axis: episodes, data, parameters
*Directions absorbed: "scaling 1M→20M+", "how quickly does transfer saturate (<1k
episodes?)", "much larger param size", "scale to 100M data points / 20M params".*

**Why it carries.** The strongest scientific objection to RQ1 is *capacity*: "your
mixture equals the max only because a 1M-param encoder cannot hold two graphs." Killing
that objection converts an observation into a law. It also converts the state-doc scaling
TODO from GFM-claim armor into a *result*.

**Hypothesis.** Transfer is a coverage phenomenon, not an accumulation phenomenon, so it
saturates early in all three budget axes and the `max` rule is invariant to capacity.
**Predictions:** (a) target transfer within ~5% of final by <1k episodes; (b) 10× params
on all-8 leaves per-target `max` structure intact; (c) within-source data subsampling
(10/25/50/100%) flattens well before 100%. **Kill:** at 10× capacity the mixture exceeds
every component on some target → composition becomes capacity-dependent. *That is still
a paper, but a different one* — say so up front and pre-register the reading.

**Cost:** (a) is nearly free (checkpoint sweep on existing runs); (c) ≈1 GPU-day (E4);
(b) is the one multi-day run (E14). Run (a) and (c) first — if saturation is sharp, (b)
becomes cheap to justify and cheap to interpret.

### T1.2 **[M]** Gradient-conflict + representation analysis
*Directions absorbed: "gradient conflict analysis / CKA", "add topology features then
inspect topology and cross-bio-topology weights".*

**Why it carries.** It supplies the *mechanism for the asymmetry* that is currently our
most surprising and least explained result: interleaving rescues multi-graph but not
multi-objective. Right now that asymmetry is an observation; with gradient geometry it
becomes an explanation, and explanation is what separates a workshop paper from a main
track one. Highest value-per-GPU-hour item in the entire list.

**Hypothesis.** Graphs are near-orthogonal tasks; objectives are conflicting tasks.
**Predictions:** per-graph gradient cosine ≥ 0 (near-orthogonal to aligned) while
per-objective cosine < 0 at the shared encoder; CKA between single-objective and mixed
checkpoints shows the mixed model drifting away from the aligned objective's
representation rather than adding a subspace. **Kill:** cosines are indistinguishable
across the two groupings → the asymmetry is not gradient-geometric and we drop the
mechanistic claim, keeping only the phenomenology.

**Cost:** logging hooks + short instrumented reruns; CKA is analysis-only on existing
checkpoints. Days, not GPU-days.

**Caveat to design around now:** cosine must be measured at a fixed shared layer with
matched batch composition, otherwise per-graph vs per-objective cosines are not
comparable. Decide the probe point before running.

### T1.3 **[I+M]** Is the SSL objective its own transferability predictor?
*Directions absorbed: "SSL indicator: does SSL performance / its gradient / activations
indicate downstream performance?", "can we predict cross-domain transfer for all tasks
with NM? how correlated is NM with downstream?".*

**Why it carries.** This is the *applied payoff* of RQ2 and the thing practitioners
actually want: a cheap in-training signal that tells you whether pretraining is working
before you run a downstream eval. It also strengthens C2 by adding a second, independent
predictor family (in-training) alongside the data-side divergence.

**Hypothesis.** Held-out NM AUC on a target graph is a sufficient statistic for that
target's downstream frozen performance.
**Predictions:** within-target rank correlation between held-out NM and downstream
node-classification/regression across the 8×8 matrix is high (|ρ|>0.7); it is *not*
predictive for LP after the evaluator repair (LP is an NM main effect, so this one may be
circular — state that). **Kill:** correlation is target-specific or near zero → demote to
"SSL loss is not a model-selection signal", which is a useful negative but not a headline.

**Cost:** mostly analysis over existing eval artifacts + one instrumented rerun to dump
per-episode SSL metrics. Cheap. **Do this early** — it is cheap enough that it should not
be sequenced behind anything.

### T1.4 **[M defensive / I offensive]** Topology: the nhop confound, and a task that actually needs topology
*Directions absorbed: "why does adding topological information do worse than bios?",
"vary masking per episode", "topology-only test with n_hop=4 and limit≫100", "add a
downstream task requiring topology (real CSS task)", "cascade / hashtag prediction",
"LLM comparison — Opus as an unbeatable ceiling on bios".*

**Why it carries.** Two separate jobs that split cleanly across the two papers:

1. **Defensive — Paper M (must do).** Our nhop=1 sampling made topology nearly unusable *by
   construction*. This is the single most attackable point in the whole program and we
   already know it. Reviewer 2 writes "your conclusion is an artifact of the receptive
   field" and the mechanism section dies. Rerun the feature-forcing arms at nhop≥2 with a
   large neighbor limit, plus a structural-featurization arm (degree/LapPE/RWSE inputs).
2. **Offensive — Paper I (should do).** A new CSS task plus the bio-only LLM ceiling is a
   *social-lane* contribution, and it is the thing that makes Paper I more than a transfer
   matrix. The LLM comparison is the real threat, and naming it first
   is the strongest move available. If bio text carries the signal, a frozen LLM reader is
   a ceiling we cannot beat, and the honest response is not to avoid the comparison — it
   is to *run it* and use it to motivate a task where the graph is not optional. Cascade
   size / diffusion participation and hashtag adoption are both genuine CSS tasks and both
   plausibly topology-bound.

**Hypothesis.** Topology is not unreadable — it is *uninformative for the tasks we
measured*. **Predictions:** at nhop≥2 with structural inputs, transfer on current tasks is
unchanged (feature-dominance survives the control), *but* on a cascade/adoption task
structural inputs and larger receptive fields produce a real gain over a bio-only LLM
reader. **Kill:** nhop≥2 alone recovers topology-driven transfer on existing tasks → B is
substantially an artifact and must be rewritten as "topology needs ≥2 hops", which is a
much weaker but still publishable finding.

**Cost:** nhop rerun ≈4–6 rungs (E9). Cascade/hashtag task is data engineering, ~1 week —
it is the largest genuinely new build in this doc, and the only one worth the money.
LLM/GTE-bio baseline is eval-only.

### T1.5 **[M]** Sampling coverage as the mechanism behind saturation
*Directions absorbed: "how are we exploring the graph as we sample more (supernodes)?",
"how much of the graph have we explored by episode 40k?", "sampling: uniform sampling
creates clusters and gaps, power law, importance sampling", "why don't we reach 100 AUC".*

**Why it carries.** Reframed, this stops being a tuning knob and becomes the measurement
that explains T1.1: *if effective node/edge coverage saturates early and concentrates on
hubs, then both early transfer saturation and the max-composition rule have a common
cause.* That is a mechanism linking the sampler to the law. Framed as "we improved the
sampler and got +2 AUC," it is worth nothing at these venues.

**Hypothesis.** Episode sampling is hub-concentrated; effective coverage saturates long
before the loss does, and the un-covered tail is where the residual error lives.
**Predictions:** unique-node/edge coverage curves flatten well before 40k episodes;
coverage-weighted degree distribution is far more hub-skewed than the graph's; per-node
downstream error correlates with sampling frequency. **Kill:** coverage is still climbing
at 40k and error is uncorrelated with visit frequency → drop the mechanism link, keep a
one-paragraph sampler-diagnostics note.

**Split.** Paper M gets the coverage→saturation mechanism link. Paper I gets at most one
descriptive paragraph ("our sampler sees mostly hubs; here is what that means for your
corpus") in limitations — no mechanism claim.

**Note.** The importance-sampling *remedy* is only worth running if the diagnostic
confirms the pathology, and even then it belongs in an appendix — the diagnostic is the
contribution, the fix is engineering. (Prior evidence is already discouraging: the
cross-source-probability sweep found within-source sampling best and remedy #4 unhelpful.)

### T1.6 **[I+M]** Put the ladder on the downstream tasks
*Directions absorbed: "test the graph ladder models on the downstream tasks", "graph
ladder results for sequential training".*

**Why it carries.** RQ1 is currently established mostly on the pretext metric. A
composition rule that only holds for NM-on-held-out-edges is a curiosity; one that holds
for bot detection, suspension, and political leaning is a law. This is the cheapest
possible upgrade to the headline: the checkpoints exist, only evals are missing.

**Hypothesis.** The staircase and the entry-jump reproduce on downstream tasks.
**Prediction:** per-task jumps align with the same donor entries as the pretext ladder.
**Kill:** downstream ladders are flat → the composition rule is pretext-specific, which
substantially weakens *both* papers and should be found out *now*, before anything else is
built on top of it.

**This is the highest-priority item in the entire document, and the one gate both papers
share.** It is eval-only and it can falsify Paper M's rule and Paper I's recipe at the same
time. Run it first. Paper I reports it as "adding graphs stops helping the tasks you care
about"; Paper M reports it as the rule holding across task families.

Sequential (catastrophic-forgetting) ladders are already a finished result — keep as one
figure motivating why interleaving is the baseline, not as an open direction.

---

## 3. Tier 2 — required rigor, not contributions

These do not go in the contribution list. They go in tables, and their absence is a
rejection reason. Bundle them; do not sequence them individually.

| Direction (from the list) | Venue | Role | Where it lands |
|---|---|---|---|
| "Train PRODIGY from scratch on all downstream tasks" | **I** | in-protocol supervised baseline to beat | floors table (E2) |
| Supervised skylines (BotRGCN/RGT/SeBot; ideology & suspension priors) | **I** | the comparison a CSS reviewer demands | main results table |
| "Compare PRODIGY with a normal GATv2/GraphSAGE encoder" | **I+M** | is the meta-learning machinery earning its keep? | floors table |
| "Compare another model" | **I** | external comparability | pick **one** generic GFM (AnyGraph/OpenGraph-style) — not a survey |
| "Frozen vs. hot embeddings" | **I+M** | protocol characterization | protocol section, both papers |
| "Finetuning" ×3 (all-experiments, beat-the-specialist, harder-task-than-others) | **I** | protocol bridge to the supervised literature | see below |
| "Why don't we get 100% on NM/POL/SUS/BOT — what are the specific cases" | **I** | error analysis, per dataset | limitations/analysis section; feeds T1.5 |
| Untrained-encoder + raw-feature + SGC/NAFS floors (E2) | **I+M** | rules out "it's just architecture" | every results table, both papers |
| Seeds / MDE table (E1), LP hardening incl. HeaRT negatives (E3) | **I+M** | every number we quote is 1 seed | methods, both papers |
| "Multi-SSL: switch every 10 episodes; unbalanced episodes" + "sum the losses instead of rotating (weighted)" | **M** | completes the 2×2 vs ControlG | RQ4 section (E10) |
| "Swap GraphSAGE for GIN" / "when does GNN X beat GNN Y" | **M** | architecture invariance of the rule | one robustness row, reframed |

**Note the asymmetry:** almost all baseline/skyline/finetuning debt belongs to **Paper I**
— that is the paper whose readers know the supervised numbers by heart. Paper M needs
floors (untrained, raw-feature, SGC) but not a social-media leaderboard.

**On finetuning — resolve this once, now, and note it is a Paper I problem.** The
all-frozen protocol makes our numbers non-comparable to most of the literature and makes
our models look weak — but that only bites in front of the social-media audience, where
supervised BotRGCN/RGT numbers are common knowledge. The resolution is not to switch
everything: **keep frozen/in-context as the headline protocol in both papers (it is what
makes the transfer claim clean — nothing can be relearned at eval time), and add a single
fine-tuned arm on the pivotal comparison** (best specialist vs all-8) *in Paper I only*.
That answers "can finetuning beat the specialist" and the comparability objection with one
experiment instead of a full re-run. Say explicitly in the protocol section that frozen
eval is a *lower bound*. Paper M does not need this arm at all: there, frozen eval is a
methodological virtue, not a handicap.

**On GIN/architecture:** only interesting as "the composition rule is not an artifact of
GraphSAGE." One swap, one row, in the appendix. "When does GNN X beat GNN Y" as a
standalone question is a different paper and a crowded one.

---

## 4. Tier 3 — park, cut, or repurpose

| Direction | Verdict |
|---|---|
| **"Reproduce the 3-way SSL result with another model"** | **Dead as written.** The result being replicated does not exist: the 2026-07-23 rescore showed the LP synergy was an evaluator artifact; on valid metrics LP is an NM main effect and every added objective lowers it. Do not spend a GPU-hour replicating a retracted finding. |
| **"Replicate emergent sLP on an OGB graph"** | Same — repurpose the OGB engineering into **E11 (an alien donor for the composition rule)**, which tests external validity of A and preempts "all your graphs are Twitter." Same data work, live claim instead of a dead one. |
| **"Injecting downstream tasks in pretraining for ICL"** | Whole separate paper (supervised multi-task pretraining). Scope creep; park. |
| **"Joint graphs over disjoint graphs"** | Needs cross-graph edges we do not have; building them is a research project in entity resolution. Park with a note. |
| **"Add task: cascade / hashtag prediction"** | **Promoted** into T1.4 — only justified as the topology-requiring task, not as generic task-count inflation. |
| **"Add topology features, inspect weights, are cross-bio-topology weights ~0?"** | **Promoted** into T1.2 as the representation-analysis half. On its own, weight-inspection is weak evidence (weights are not attribution); pair it with CKA/ablation or drop it. |
| **"Scaling: 1M→20M+"** as an independent line | Promoted into T1.1 as *capacity control*, not as a scaling-law paper. We do not have the arms for a real scaling law (n=8 graphs) and should not claim one. |

---

## 5. What to run, in order — two lanes

### Lane 0 — shared gate (weeks 1–3, before either lane commits)

0a. **T1.6 — ladder on downstream tasks.** Eval-only. Can falsify Paper M's rule *and*
    Paper I's recipe. Nothing in either lane matters until this returns.
0b. **P0 rigor: E1 seeds + MDE table, E2 floors, E3 LP hardening.** Every number we
    currently quote is 1 seed. Both papers are blocked on this.
0c. **T1.3 SSL-as-predictor** — cheap, analysis-mostly, feeds both. Start immediately in
    parallel; it is not on anyone's critical path but pays into both.

**Decision point at end of week 3.** If 0a+0b are clean → commit to ICWSM Sep 15. If they
slip or the downstream ladder is flat → drop to the ICWSM Jan 15 round and re-plan; do not
submit a 1-seed paper.

### Lane I — ICWSM, deadline Sep 15, 2026 (~7 weeks)

Everything here is eval-only or data engineering. **No new pretraining runs** — that is
what makes the deadline plausible.

1. **E8 source-selection payoff** — predicted-best k=2–3 subsets vs all-8. The paper's
   punchline. ~1 GPU-day via the subset knob; pre-register the prediction with a repo hash
   *before* training.
2. **Tier 2 baseline/skyline bundle** + one fine-tuned arm + protocol-bridge write-up.
3. **Evaluator forensics section** — the audit, the repair, the inversion; released fix.
4. **Corpus + task-suite documentation** — catalog, splits, eval harness, release plan.
5. **T1.4-offensive: cascade/hashtag task + bio-only LLM ceiling.** *Start the data
   engineering in week 1* — it is CPU/data-bound, not GPU-bound, and it is the only Lane I
   item that can miss the deadline. If it does miss, it moves to the Jan round or becomes
   Paper I's follow-up; the paper stands without it.
6. **Error analysis** ("why not 100%") per dataset — cheap, and it is the kind of thing
   ICWSM reviewers actually read.

### Lane M — ICML 2027 (≈Jan) / NeurIPS 2027 (≈May)

Starts in parallel where GPU-free, but takes the GPUs from mid-September.

7. **E5 functional-form fit + E6 predictor-at-scale** — analysis-only, near-zero GPU;
   start now, they are the intellectual core and the longest analysis lead time.
8. **T1.1(a,c) saturation in episodes and data** (E4) — cheap; its answer sets how much
   capacity work T1.1(b) needs.
9. **T1.4-defensive: nhop≥2 + structural-featurization rerun** (E9) — closes our known
   weakest point.
10. **T1.2 gradient/CKA** — instrumented reruns; the mechanism upgrade.
11. **T1.5 coverage diagnostics** — analysis on existing sampling logs.
12. **E7 interventional dose-response** — the causal differentiator.
13. **E10 objective 2×2 completion** vs ControlG.
14. **T1.1(b) 10× capacity + E11 alien donor** — the invariance pair; only after 7–10 hold
    up. These are the two most expensive items in the program and the last to commit to.

### Timing and preprints

Per the scoop watchlist the mechanism (B) and objective-dilution windows are actively
closing (three convergent papers Feb–Jun 2026). Consequences:

- Post **Paper I's preprint at ICWSM submission** (mid-Sep) — that timestamps the corpus,
  the donor structure, and the evaluator fix.
- If Lane M items 9–12 are done but the full paper is not by ~Nov, **post the mechanism
  section as a standalone preprint** rather than holding it for a venue. An arXiv date on
  the interventional mechanism is worth more than which of ICML/NeurIPS it lands in.
- **Skip ICLR 2027** (≈late Sep 2026). Attempting it collides head-on with ICWSM and would
  force Paper M out before its invariance arms exist — which is exactly the paper the
  reviewers would reject.
- ICML/NeurIPS 2027 dates above are approximate — **verify before planning against them.**
  Only the ICWSM (Sep 15 / Jan 15) and WSDM dates are confirmed by the Jul-25 sweep.

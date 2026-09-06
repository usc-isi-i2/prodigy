# Query context versus support context: from cases to complete replay

6 September 2026. **Exploratory mechanism localization with a three-seed replication.**
This explains part of the computation behind a target-specific donor gap; it
does not establish which property of pretraining data caused that computation.

The [natural-support follow-up](FINDINGS_NATURAL_SUPPORT.md) is now complete:
unmodified support replacements confirm greater political sensitivity for
Hong Kong, with all queries fixed. Support-only held-out selection helps Hong
Kong's political NLL in every seed/stream but usually worsens Ukraine's.

## Main result

On COVID Political, removing background message-passing edges **only from the
support examples** improves Hong Kong's AUC in every one of three existing
initialization seeds, on both fixed episode streams: **+.0465 to +.1110**.
Ukraine's change is between −.0019 and +.0002. Applying this same intervention
to both sources reduces their AUC gap by **52–71%** across the six comparisons.

The query inputs and their representations both before and after the metagraph
remain **bit-for-bit unchanged**. Decoding the original query representations
against the altered label representations reproduces every altered logit
exactly, in all 384 checked batches. This localizes the effect to the
support → label-representation → query-score pathway of the fixed model.

The intervention retains all graph-selected support members and their features,
pooling edges, support labels, and all query input. It removes only support
background message-passing edges. It is **not** a graph-free comparison, natural
support resampling, or a training-data intervention. These results are about
agreement with dataset labels, not independently verified political attributes.

## 1. What the full example-level replay contains

The preceding [24-case review](FINDINGS_EXAMPLES.md) motivated this experiment.
We did not estimate prevalence from those selected cases. We replayed all nine
historical singleton checkpoints, all five classification targets, both fixed
episode streams, and all eight conditions: **720 complete cells**. Each cell
uses 128 episodes. Each target/stream has 3,072 queries for COVID Political and
TwiBot20, 1,024 for Facebook Page Reference, and 256 for Election and Ukraine
Suspended. These are query occurrences, not necessarily unique nodes.

Conditions are baseline; query/support/both context-feature duplication;
query/support/both background-edge removal; and query-center-feature zeroing.
Feature duplication replaces a context node's features with its own subgraph
center's features, retaining membership and topology. The roles are factored
at the pre-metagraph representations, using the actual downstream model.

Validity: 2,880 cached baseline batches are bit exact; all 6,750 direct
whole-forward/factorization/localization checks have zero error; all weights
remain unchanged. Independently saved metrics and provenance match for 180
reference cells. Every result is retained, including newly caused errors and
null/reversed effects. Cohort partitions and per-class counts conserve totals.
The complete grid uses one historical training seed; the follow-up below uses
all three pre-existing production-policy seeds of two sources.

## 2. Where Ukraine/COVID's political advantage occurs

For the historical checkpoints, nearly all excess correct answers over Hong
Kong occur among queries that have sampled context:

| Contrast | Stream | Extra correct: all queries | Extra correct: with context | Fraction of total gap |
|---|---|---:|---:|---:|
| Ukraine − Hong Kong | Original | 656 | 617 | 94.1% |
| Ukraine − Hong Kong | Fresh | 613 | 609 | 99.3% |
| COVID − Hong Kong | Original | 628 | 624 | 99.4% |
| COVID − Hong Kong | Fresh | 570 | 589 | 103.3% |

The last value exceeds 100% because COVID makes 19 fewer correct predictions
than Hong Kong among isolated queries. This is an exact decomposition of the
accuracy gap, not a fraction of AUC explained or a causal mediation percentage.
The context cohorts contain 2,079/2,103 queries; the isolated cohorts 993/969.
Their class balances differ, so cohort accuracy alone is not comparable evidence
of task difficulty. Within the context cohort, Ukraine's macro-class recall
also exceeds Hong Kong's: .9334 versus .6912 / .9450 versus .7164.

Replacing only Ukraine's query-context features reduces political AUC by
.0737/.0648 and accuracy by .1045/.1048. Its query-edge removal has little AUC
effect (+.0024/−.0006). This supports a specific distinction: graph-selected
neighbor features are useful even when the original background edges are not
needed by that checkpoint on this task. The declared directional expectation
replicated across streams; the study remains exploratory because the prior
case review and earlier interventions informed it.

## 3. Support-side processing accounts for much of the source gap

Historical political AUC, original / fresh:

| Source | Baseline | Query edges removed | Support edges removed | Both removed |
|---|---|---|---|---|
| Ukraine | .9464 / .9572 | .9488 / .9566 | .9461 / .9563 | .9497 / .9574 |
| Hong Kong | .7489 / .7932 | .6764 / .7265 | .8552 / .8853 | .8646 / .8918 |
| COVID | .9335 / .9470 | .9274 / .9388 | .9340 / .9450 | .9310 / .9405 |

The historical Ukraine–Hong Kong AUC gap shrinks by 53.9%/56.7% under the
same support-only edge intervention. Hong Kong's query-only edge removal instead
**worsens AUC**, even though accuracy increases. Its changes in ranking and
decision threshold/calibration must not be conflated.

Support-edge removal is not uniformly beneficial to Hong Kong's queries:
it fixes 478/414 errors but introduces 219/259 new ones. Among queries with
context, it fixes 393/352 and introduces 97/130; among isolated queries, it
fixes 85/62 but introduces 122/129. These paired counts explain why a dramatic
individual rescue is insufficient and why the effect is cohort-dependent.

### Three-seed check and exact computational pathway

We then evaluated **all six existing lowest-ID/sorted-role controls** from the
completed member-policy experiment: Ukraine/Hong Kong × seeds 0, 1, 2, at the
pre-existing 2,500-update checkpoints, with no new training or checkpoint
selection. These are different checkpoints from the historical seed-0 models
above and must not be merged into one repeated-run estimate.

| Source / seed | Original baseline → support-edge removal | Fresh baseline → support-edge removal |
|---|---|---|
| Hong Kong / 0 | .8202 → .9065 | .8590 → .9154 |
| Hong Kong / 1 | .7348 → .8458 | .7715 → .8670 |
| Hong Kong / 2 | .8448 → .9120 | .8752 → .9217 |
| Ukraine / 0 | .9416 → .9417 | .9543 → .9528 |
| Ukraine / 1 | .9434 → .9417 | .9571 → .9557 |
| Ukraine / 2 | .9414 → .9405 | .9519 → .9500 |

All 12 baselines reproduce the independently saved evaluation; full input and
weight hashes match. All 384 batch traces verify unchanged query vectors
before and after the metagraph and exact reconstruction from changed label
vectors. There is one metagraph layer and no final reverse layer in these
models; the executable check guards this architectural restriction.

The label vectors change for both sources, but Ukraine's accuracy ranking is
robust to that particular change while Hong Kong's improves. A representation
change by itself is therefore not an explanation of harm. The established
result is its direction at the final output, with the pathway localized.
Earlier CPU training has execution variability; these are fixed-checkpoint
input interventions across three initialization seeds, not a controlled estimate
of training-source effects or six independent seeds.

## 4. The Facebook anecdotes are real, but not a general remedy

For Ukraine's Facebook classifier, replacing only query-context features fixes
**77 errors and causes 74** on the original stream, but fixes **71 and causes
83** on the fresh stream. The spectacular reporter example was real; the
population-level accuracy gain is not replicated (+.0029/−.0117).
Replacing only support-context features fixes 70/65 but causes 83/73 errors.

Across all eight foreign donors, mean AUC changes are negative on both streams:
query-feature duplication −.0118/−.0062; support-feature duplication
−.0162/−.0106. Do not recommend indiscriminate context removal from this audit.
Facebook's large raw-description-probe advantage remains a separate diagnostic
gap; correcting selected examples does not close it.

## 5. TwiBot confirms that context has genuine predictive value

Removing query edges lowers AUC for **all eight foreign donors on both streams**
(mean −.0544/−.0556); support-edge removal does too (−.0466/−.0534).
Query-feature duplication (−.0436/−.0325) and support-feature duplication
(−.0350/−.0288) also harm every foreign donor on both streams. Thus the
incoming-degree probe's superiority does not mean the models' graph processing
is globally useless. Degree is sampled retweet-graph role, not full-network
popularity, and isolated accounts include both dataset classes.

Election likewise depends strongly on query-context features. Query/support
effects can interact: Hong Kong's original AUC is .9333; query-edge removal
gives .7122, support-edge removal .9701, and removing both .9820. The joint
effect is not the sum of the individual effects. The exported full interaction
table retains every source, target, and stream. No comparable broad rescue was
established for Ukraine Suspended.

## Research interpretation and remaining test

An example's success depends on the **query, its sampled context, the support
episode, and the trained computation together**. We now have a replicated
fixed-model pathway that explains a substantial, target-specific donor gap:
Hong Kong's support representations induce less useful political label
representations when background message passing is active. Query-only
distribution distance cannot by itself identify this support-side mechanism.

This does not identify why Hong Kong pretraining learns it, why Ukraine/COVID
are better overall, or a novel general architecture. It does not establish
that natural support replacement has the same effect as deleting support
edges. Next discriminating tests should use valid resampled supports for fixed
queries, test whether an outcome-blind support-compatibility score predicts
failures on new episodes, and examine exact training episodes/objective
gradients for the source-dependent behavior. A second architecture/unseen-task
validation is needed before a general ICLR claim. Pair/LOO models are not
covered by this role-factorial audit.

## Evidence and reproduction

- Complete raw compact receipts: `data/role_context_replay/`; derived cells,
  cohorts, per-class counts, source-gap decomposition, and interactions:
  `data/role_context_*.{csv,json}`.
- Three-seed receipts: `data/support_path_seeds/`; derived paired effects,
  source-gap reductions, and validation: `data/support_path_*.{csv,json}`.
- Run `analyze_role_context` as a module with `--input <data/role_context_replay>`
  and run `analyze_support_path` as a module. Both independently validate stored
  reference scores, full grids, and provenance before writing summaries.
- Four role-analysis tests and two support-path-analysis tests pass, alongside
  seven runtime role/intervention tests. Full real-data validation also passes.
- Tucker runtime: `/dataMeR1/phil/gfm/prodigy-mechanisms-role`, revisions
  `b587a046` (720-cell replay) and `030ea28c` (seed/path check). Output directories
  `log/role_context_20260906/` and `log/support_path_seeds_20260906/`, both complete.
  Saved per-query tensors remain there; no raw profile texts are newly published.
- Local branch `codex/target-performance-mechanisms`, worktree
  `/Users/philipp/projects/gfm/prodigy-mechanisms`. Private Git transport only;
  CPU-only replay, no production model changes, no user-job interruption.

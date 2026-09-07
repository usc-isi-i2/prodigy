# Individual successes and failures: first matched case review

2026-09-06. **Exploratory qualitative audit, not a representative performance estimate.**

Follow-up now complete: [FINDINGS_ROLE_CONTEXT.md](FINDINGS_ROLE_CONTEXT.md)
tests these hypotheses across 720 cells and checks the political support-side
mechanism on all three existing Ukraine/Hong Kong seeds. It also shows why
the selected Facebook rescues do not justify blanket context removal.

The missing piece was reading what an example actually contains. I inspected
24 selected query cases across COVID Political, Facebook Page Reference, and
TwiBot20, and examined the actual support texts and sampled neighborhoods for
representative cases. This gives concrete examples of helpful query context,
harmful query context, and harmful support context. These are different effects;
none alone explains why a pretraining source wins on average.

## What is verified

Two cases from each of four strata per target: Ukraine right / Hong Kong wrong;
the reverse; both wrong / raw-center ridge right; both right / raw-center ridge
wrong. Selection was the first eligible query in fixed batch/query order, with
distinct episodes and query identities **within each stratum**, before reading
its text. Cases across strata may share an episode. Most selected cases come
from the first batch; they are not 24 independent episodes or random examples.
COVID is also evaluated on every selected case, but did not drive selection.

The three models are the historical singleton, seed-0, step-2500 checkpoints.
This is not yet a qualitative audit of every pair/LOO model or training example.
All 96 cached batches passed full input hashes, exact graph-feature identity,
and center-label mapping checks. Facebook/TwiBot text was joined through user
identity → bio hash → embedding shard, whose float32 bytes matched the graph.
COVID Political CSV row identities and all row labels matched the artifact.

All 12 reconstructed model/batch baselines matched saved logits **bit exactly**.
We then ran 360 localized forwards: 24 cases × 3 models × 5 changes. The changes
touch either the selected query or its same-episode supports, never their labels:

- Replace context-node features with that subgraph's center features, retaining
  sampled members, topology, and synthetic nodes.
- Zero only the selected query's center features.
- Remove only the selected query's background message-passing edges.
- Replace context features only in the 20 support subgraphs of that episode.
- Remove background edges only in those support subgraphs.

These are fixed-model interventions, not claims about retraining or natural
counterfactual graphs. In particular, removing background edges **retains the
graph-selected nodes available to pooling**. Feature duplication and zeroing
can also create atypical inputs. Reported percentages below are the model's
uncalibrated softmax score for the dataset label, not the probability that a
person truly has that attribute. Raw probes fit only the 20 episode supports.
Facebook predictions are **two-way comparisons between sampled categories**,
not a simultaneous 30-way classifier.

## 1. Helpful query context: a vague bio and an explicit political bio

**COVID Political, batch 0, query 1, graph row 7229.** The complete query bio is
“all of my opinions are based on my personal experiences”; its dataset label
is conservative. Raw-bio ridge predicts the other class. Its 10 sampled
neighbors include several explicit Trump/MAGA/conservative self-descriptions.
Ukraine is correct at **99.65%**. Replacing only this query's context features
with its own bio reduces that to **22.20%**; zeroing the query bio leaves
**99.78%**, and removing only its background edges leaves **99.08%**.

This example's successful prediction depends on the query's neighborhood
features; those features remain useful without background message passing.
Hong Kong is also initially correct, but its score drops to **4.17%** when
only the query edges are removed. Same observed answer, different dependence.

**Batch 0, query 10, row 60598.** The query explicitly describes being a Texas
liberal in a red environment; all 20 sampled neighbors have the same
non-conservative dataset label, and many descriptions contain liberal/resistance
cues. Ukraine scores the annotated class at **84.21%**, Hong Kong at **7.43%**.
Replacing this query's context features makes Ukraine wrong (**2.45%**).
Removing only query background edges makes Hong Kong correct (**99.41%**),
while preserving all sampled text and support inputs. The failure is not simply
that Hong Kong lacks relevant examples or useful query text.

## 2. Harmful query context: a reporter classified as a community organization

**Facebook, batch 1, query 2, row 15827.** Query description:

> Reporter, Host & TV Personality. Based in Southwest Florida.

The episode compares NEWS_PERSONALITY with COMMUNITY_ORGANIZATION. The closest
support by raw-bio cosine is a correctly annotated reporter/anchor (cosine
**0.737**); another reporter support scores **0.724**. Raw-bio ridge is correct.
The query has nine sampled neighbors. Its direct neighbor describes Ukraine
news/foreign policy, while the second-hop context includes community,
political-organization, NGO, and other page descriptions.

Ukraine scores NEWS_PERSONALITY at **13.40%** (wrong); Hong Kong **86.03%** and
COVID **63.99%** are correct. Replacing **only this query's** context features
with its own description raises Ukraine to **99.93%**. Removing only its
background edges barely helps (**15.63%**); changing only support-context
features makes it worse (**5.31%**).

Thus the harmful input is localized to this query's contextual representation.
It is not a lack of semantically appropriate support bios. This does not prove
which particular neighbor or feature coordinate drives the effect.

## 3. Harmful support context: the query can remain completely unchanged

**Facebook, batch 0, query 12, row 14421.** A community media page describing
coverage of the Ogoni and Niger Delta is annotated TV_CHANNEL, versus UNIVERSITY
in this episode. The only sampled neighbor is another media page. The three
closest raw-bio supports are all TV_CHANNEL (cosines **0.565, 0.558, 0.557**),
and raw-bio ridge predicts TV_CHANNEL.

All three full models are wrong: Ukraine **12.17%**, Hong Kong **10.97%**,
COVID **0.59%** for the annotated class. Replacing **only support-context**
features makes Ukraine correct (**67.46%**), without touching the query's
text, neighborhood, or edges. Replacing query context does not fix it
(**7.85%**); removing query edges also does not (**12.07%**).

This is an actual support-side failure, not merely a hard or out-of-distribution
query. The intervention localizes dependence but does not yet separate which
support, its encoder output, and the downstream label representation matter.

## 4. Correct is not evidence that the model understood the bio

**COVID Political, batch 0, query 15, row 34409.** A bio explicitly includes
“Supporting Biden,” with a non-conservative dataset label and **zero sampled
neighbors**. Hong Kong is correct at **93.37%**; Ukraine (**14.91%**) and COVID
(**4.73%**) are wrong. This initially looks like better text use by Hong Kong.

But zeroing this query's entire bio increases Hong Kong's score to **98.32%**.
Removing background edges **only from the supports** flips it wrong
(**27.50%**). Query-edge/context changes are exact no-ops because this query
has no sampled neighbors. So the correct answer does not establish that Hong
Kong recognized the political phrase; support-conditioned behavior can produce
the same answer even without query text. Conversely, this one case cannot show
that Hong Kong never uses bios.

## 5. TwiBot: comparable neighborhood sizes can encode opposite graph roles

Two inspected cases have almost the same sampled context size:

| Case | Dataset label | Context nodes | Center incoming / outgoing edges | Raw-bio ridge | Ukraine / Hong Kong / COVID score for label |
|---|---|---:|---:|---|---|
| b00 q12: U.S. Senator bio | human | 93 | 18 / 0 | bot | 64.44% / 16.30% / 41.57% |
| b00 q26: “Family Spurs Tmnt” | bot | 97 | 0 / 18 | human | 80.67% / 95.28% / 89.35% |

Edges in this reconstructed graph point from retweeter to retweeted account.
The first query is a recipient of sampled retweets; the second retweets
sports/media/public-figure accounts. The support-only incoming-degree probe
gets both right. The episode supports themselves contain many human-labeled
news/institutional accounts with large incoming counts, and bot-labeled
accounts with mainly outgoing edges, **but there are clear exceptions in
both classes**. Bio topic and neighborhood size alone obscure this distinction.

Ukraine's human prediction for the Senator case becomes wrong after removing
only query edges (**64.44% → 29.80%**). In contrast, the “Family Spurs Tmnt”
bot prediction survives query-edge removal (**80.67% → 78.22%**) but flips
after changing only support-context features (**1.26%**). Even a strong
structural correlate does not imply every model implements the same rule.

An isolated human-labeled query, b00 q36, has a short name-like bio and no
sampled edges. The incoming-degree probe predicts bot and Hong Kong scores
human at **1.10%**; Ukraine scores human at **89.53%**. Another isolated
human-labeled query, b00 q39, is missed by all three. Isolation is a useful
but fallible dataset cue, not an operational definition of a bot. These counts
are sampled subgraph degrees, not full-network degrees or original follow links.

## Annotation warning raised by reading the examples

The first political episode contains a conservative-labeled support whose bio
explicitly describes membership in a Democratic committee, and a
non-conservative-labeled support whose bio disparages liberals. These are
**apparent text/annotation inconsistencies**, not independently established
annotation errors. They could reflect noisy labels, temporal mismatch, or a
labeling definition not recoverable from a bio. I have not changed labels or
used my interpretations as new ground truth. TwiBot bot/human status likewise
cannot be established by reading a self-description.

## What this changes in the research plan

The unit of explanation must include **query + sampled context + support
episode + trained model**. A global source/target distance or query-only
difficulty score cannot distinguish the concrete support-side and query-side
effects above. These examples also contradict a blanket story that UKR/COVID
are better because they simply use more context: that context helps some
queries and hurts others, and Hong Kong can win the reverse cases.

Next, quantify these example-derived hypotheses on all cached queries and the
second episode stream: context availability and graph role; semantic agreement
between center/context/supports; fixed-query sensitivity to valid resampled
support sets; and documented annotation provenance. Retain matched node/episode
dependence in uncertainty estimates. The 24 selected cases and their large
score flips must not be reported as frequencies or average treatment effects.
Extend to pair/LOO models and exact training episodes after defining this audit;
the present cases do not causally explain the donor advantage.

## Evidence and reproducibility

- Compact numeric case records: [individual_examples.json](data/individual_examples.json).
- Full private text/identity records are outside version control at
  `/Users/philipp/projects/gfm/paper/evidence/individual_examples_2026-09-06/`.
  SHA-256 hashes and exact local paths are in the compact records.
- Tucker outputs: `/dataMeR1/phil/gfm/prodigy-mechanisms-numerics/log/example_audit_20260906/`
  and `log/example_texts_20260906/` in the same worktree; both have DONE markers.
- Runtime revision `9fe8a26f`; four focused intervention/selection tests pass.
  Exact checkpoint paths and all original batch hashes remain in the records.
- Local branch/worktree: `codex/target-performance-mechanisms`,
  `/Users/philipp/projects/gfm/prodigy-mechanisms`. Code transport was private
  Git only. No public push, production-code changes, or user-job interruption.

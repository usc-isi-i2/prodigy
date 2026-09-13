# HK→HK NM: failure phenotypes inside the large residual

8 September 2026. Cached-only analysis of the 43,960 canonical HK errors for
which the learned pre-M mean cosine, a valid 30-way raw neighborhood-mean
comparison, and sampled-node Jaccard do not rank the assigned class first.
These outcome-conditioned phenotypes organize follow-up tests; they are not a
causal partition.

## Four distinct populations

### 1. Graph-valid alternatives and conflicting relational roles

14,853/43,960 residual predictions (33.79%) are another recorded neighbor of
the query: 2,934 test, 9,751 train, and 2,168 validation edges. They remain wrong
under the assigned-anchor benchmark, and train/validation edges are not held-out
test positives. Still, they show that one-third of the residual is not simply a
prediction of an unrelated candidate.

The query center also occurs inside a rival class's sampled support subgraph in
80.34% of residual errors, compared with 60.65% of correct occurrences. This is
correlated with degree and repeated membership. It does not prove that this
overlap caused the prediction, because global node identity is not passed across
independently encoded subgraphs.

A representative alternative-neighbor case contains an AFP query, Reuters and
other global-news supports in competing classes, and the query itself as a
support occurrence for another class. The episode's exclusive signs assign
different roles to near-identical or identical news entities. This is a concrete
objective-ambiguity mode.

### 2. Persistent, class-wide semantic aliases

31,654 residual errors (72.01%) occur in an anchor class where all four queries
are wrong. Residual queries have median graph degree 1,546 and median canonical
frequency 119, versus 654 and 34 for correct occurrences. The full residual is
therefore dominated by systematic query/class difficulty rather than isolated
last-layer flips.

One representative persistent, no-recorded-edge error has true and predicted
supports that all describe Hong Kong democracy activism. The model's chosen
class is semantically coherent but is not the sampled retweet-neighbor anchor;
the assigned class ranks 18th with a −3.11 native margin. Here the observable
input does not make exact relational membership easy to distinguish from a
semantic alias. This is evidence for a task-information mismatch, not proof that
the complete sampled topology lacks a usable signal.

### 3. Source-sensitive cases exist, but are a minority of the residual

The Ukraine model is correct on 2,586 residual errors (5.88%). Both models are
wrong on the other 41,374 (94.12%); in 35,128 cases they choose different wrong
classes. Thus most residual errors are shared difficulty without a common wrong
attractor. A source-training explanation must account for improvements inside a
large hard population, not assume every HK failure is uniquely induced by HK
training.

A representative source-sensitive case is a generic Japanese-language query.
The HK model ranks the assigned class second while Ukraine selects it. The true
and rival supports are both mixtures of Japanese and Hong Kong activist profiles,
so this case localizes a source-dependent boundary but does not reveal why that
boundary was learned.

### 4. Episode-sensitive repeated queries

3,895 residual errors (8.86%) come from queries that are mixed or mostly correct
elsewhere, while 40,065 (91.14%) come from queries that are always or mostly
wrong. A highly repeated activist account provides a concrete mixed-outcome
example: its anchor assignment, supports and competing classes change across
episodes, and the inspected occurrence ranks truth 13th. Such cases can diagnose
support/candidate sensitivity. They should not be mixed with the persistent
population when testing a repair.

## What these modes imply

The residual is not mainly missing text: the mean fraction of zero-feature nodes
inside query subgraphs is 13.58%, versus 14.36% for correct occurrences. Nor is
undefined raw geometry distinctive: 52.43% of residual errors versus 53.64% of
correct occurrences lack a valid 30-way raw comparison. Larger sampled query
subgraphs, high degree, rival-context identity, repeated queries and class-wide
failure travel together; the present stratification does not isolate them.

The most defensible next intervention targets the explicit contradiction in the
episode metagraph: a support assigned positive to one anchor can be a recorded
neighbor of a rival anchor while its rival relation is encoded as negative.
Using cached pre-M embeddings, compare the native metagraph with a version that
masks only negative support→label edges supported as positive by **training-view
adjacency**. Hold query/support embeddings, positive edges, candidate classes and
decoder fixed. Test the full 61,440-query stream and report both recoveries and
lost successes, with separate results for graph-valid-alternative, persistent
absent-edge and episode-sensitive populations. A test-view-aware variant may be
used only as a diagnostic upper bound because it is not deployable.

This directly connects a source-graph property—overlapping memberships—to a
specific model operation and a measurable outcome. If it fails, the next branch
is structural representation: the semantic-alias cases need an encoder signal
that distinguishes exact relational neighborhoods. More support resampling alone
does not address the contradiction; training already redraws members and sampled
contexts.

[Aggregate phenotypes](data/canonical_split/nm_hk_failure_phenotypes.json) ·
[parent decomposition](FINDINGS_NM_HK_FAILURE_DECOMPOSITION.md). Private rows and
representative bios remain under
`/dataMeR1/phil/gfm/error_audit/nm_hk_failure_decomposition_20260908/`.
Runtime revision `abd85403`, branch `codex/nm-hk-goal-20260908`. The successful
join and edge lookup took about six seconds on two low-priority CPU threads; no
GPU, graph loading, encoding or new sampling was used.

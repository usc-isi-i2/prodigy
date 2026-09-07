# Repeated-query contexts do not resolve the source-performance explanation

Completed 6 September 2026. This is a no-update diagnostic on actual training
inputs, not a new target benchmark, training repair, or claimed novel theorem.

## Outcome

Hong Kong has a substantial ambiguity floor when an account appears as a query
for multiple anchor-defined pseudo-classes. But its measured loss is far above
that floor, and its original context assignments confer essentially no average
advantage over context exchange. Accounts appearing only once as queries also
have much worse Hong Kong predictions than Ukraine/COVID predictions on their
respective source tasks. Thus an irreducible duplicate-query ceiling alone is
not a sufficient explanation of the observed training difficulty. This does
**not** exclude indirect damage from conflicts during learning.

The code path and full-input swaps support a more precise statement than the
earlier identity-only audit: the bound applies to a **symmetrized context
experiment**, even for this graph model. It is not a bound on an arbitrary
finite original context assignment, nor an explanation of target AUC.

## Matched input and model scope

Nine existing free-readout controls: Ukraine, Hong Kong and COVID × three seeds;
initial and 2500-update checkpoints; four actual consumed batches per model;
production training normalization and frozen metagraph normalization. Each batch
contains four 30-way episodes, three supports/four queries per class: 480 queries.
The 36 complete input hashes match the original training records, and the prior
actual-TrainerFS reconstruction remains the reference.

There are 144 input/checkpoint/normalization cells and 432 cohort rows (all,
unique-query identity, repeated-query identity). The 63,520 group-statistic rows
repeat identities across checkpoints/modes; they are not independent examples.
These are first-four-batch probes of a training stream the model has seen, not
held-out NM accuracy or full-stream statistics. Earlier full-stream identity
rates use a different scope and must not be substituted here.

## Why the symmetrization is justified—and where it stops

For a query account appearing `k` times within one episode under `k` distinct
pseudo-classes, cyclically exchange its already sampled subgraphs among those
positions. Supports, query centers, truth, label nodes and metagraph edges stay
fixed. If predictions move with those contexts, every score vector is evaluated
against each of the `k` truths once. Expected correct count is at most one in
that group; its expected total cross entropy is at least `k log(k)`.

This follows by averaging each probability vector over the distinct group
classes; their total probability is at most one. A particular original assignment
can exceed that accuracy ceiling or fall below the loss floor. The unit tests
include precisely that counterexample. No finite-input impossibility is claimed.

The inspected production path at runtime revision `b3f0c47a` is:

- `data/dataloader.py:433`: the anchor's walks produce member account IDs;
  dedicated walk/retention/role generators separate this from worker context RNG.
- `data/dataset.py:24` and `experiments/sampler.py:111`: the context sampler
  receives only the member's ID and fixed graph/fanout settings, not its anchor,
  pseudo-label or role. Recursive dictionary loading does not pass dictionary keys
  into `get_subgraph`.
- `data/covid19_twitter.py:501` and `data/augment.py:196`: these NM inputs use
  one augmentation per member, with an empty augmentation setting (= identity).
- `data/dataloader.py:961`: the collator sets roles and labels after contexts
  are sampled. Each query connects to all episode labels with the same attributes.
- `models/general_gnn.py:101` and `models/metaGNN.py:251`: the checked `S,U,M`
  model uses frozen learned label vectors for NM and no query-position encoder.
  MetaGNN ignores sequence arguments, including query truth sequences. The
  anchor IDs and other graph metadata are not used as model features.

The context API is anchor-blind, but that alone is not a proof that a finite
pseudorandom stream has independent draws. A population bound additionally
assumes conditionally exchangeable **fresh** context draws for fixed weights;
trained weights may memorize realized training contexts. The finite
symmetrization measured here does not require that population assumption.

## Input-level validation

All 144 cached-embedding suffix baselines match the saved full-model logits and
post-metagraph tensors bit for bit. One label-blind cyclic permutation per
identity/episode is then executed through the production NM suffix.

All 72 frozen-metagraph suffix permutations are bit-exact row permutations.
With training metagraph BatchNorm, the largest logit discrepancy from the
mathematical permutation is `6.68e-6`, consistent with changed reduction order.

For the first actual batch of every model, both checkpoints and modes also run
the **whole permuted PyG subgraphs**, including their original nodes, features,
edges and pooling nodes: 36 checks. Identity reconstruction of the original
batch preserves its complete hash. Whole-model baselines are bit-exact. The
largest permutation error is `1.43e-5` in logits and `1.23e-5` before the metagraph;
there are **zero unexpected argmax changes**. Encoder BatchNorm remains in
training mode, so whole-input results are not described as bit-exact.

An independent implementation rehashes all 36 inputs and explicitly enumerates
every cyclic assignment from raw saved score tensors for all 63,520 identity
groups. It also rechecks all 36 saved whole-input tensor witnesses. This is
independent enumeration of score assignments, not a full new forward for every
possible context permutation. The producer and independent verifier agree to
`1.43e-14` maximum group-statistic error, as recorded in
`data/query_exchangeability/independent_verification.json`.

## Results

Checkpoint 2500, training normalization; source means over the three seeds,
each using four fixed batches. Loss is mean cross entropy in natural-log units.

| Source | Observed accuracy | Symmetrized accuracy | Symmetrized accuracy ceiling | Observed loss | Symmetrized loss | Ambiguity floor |
|---|---:|---:|---:|---:|---:|---:|
| COVID | .65486 | .65477 | .99531 | 1.06897 | 1.07007 | .00668 |
| Ukraine | .50399 | .50472 | .98455 | 1.62754 | 1.62696 | .02169 |
| Hong Kong | .27448 | .27405 | .77708 | 2.34591 | 2.34563 | .39029 |

Hong Kong's floor is about 16.6% of its symmetrized loss; the remaining 83.4%
is above that lower bound. This arithmetic is **not a causal percentage of loss
or transfer performance explained by duplicates**. In particular, conflicts
could indirectly damage predictions on nonduplicated accounts during training.

For Hong Kong, original-minus-symmetrized loss is `+.000408`, `-.000067`,
`+.000481` across seeds. Frozen-metagraph values are `+.000474`, `-.000089`,
`+.000409`: no consistent original-context advantage. Ukraine's original
assignment is slightly worse in all seeds; COVID's is slightly better in all
seeds, by only `.000786–.001475` whole-batch loss units. This is an average
comparison, not a statement that individual predictions ignore context.

An exact decomposition of symmetrized loss is useful. For each context's
probability vector, let `M` be probability mass on the group's possible classes.
Then its mean loss over those classes equals
`log(k) - log(M) + KL(uniform(k) || probabilities-conditioned-on-those-classes)`.
The three nonnegative terms are ambiguity, insufficient mass on possible
classes, and imbalance within those classes.

| Source | Ambiguity | Mass deficit | Within-group imbalance | Total symmetrized loss |
|---|---:|---:|---:|---:|
| COVID | .00668 | 1.06187 | .00152 | 1.07007 |
| Ukraine | .02169 | 1.59856 | .00671 | 1.62696 |
| Hong Kong | .39029 | 1.90121 | .05413 | 2.34563 |

These are descriptive decompositions, not experimentally isolated causes.
For unique-query identities, `k=1`, so all loss is necessarily mass deficit.
Their mean accuracies/losses are COVID `.65750 / 1.06443`, Ukraine
`.51352 / 1.60237`, Hong Kong `.34683 / 2.15359`. The graphs and source tasks
differ; these cohorts are not content-matched counterfactuals. "Unique query"
also does not mean the identity never appears as a support.

## Implication for the research direction

The previously completed role-only training shuffle makes query-identity
ambiguity worse while improving political transfer in all three seeds and both
episode streams. Together with the present far-from-floor predictions and the
source-reversed support-label gradient test, this rules against claiming a
simple common duplicate-label explanation for Ukraine/COVID donor quality.
It does not rule out a more complex training effect of conflicting supervision.

The strongest positive downstream evidence remains model-dependent
support-context fragility, traced to changed label representations and reproduced
with natural supports. Its previous natural-support experiment used a larger
labeled pool. A necessary next control is to vary neighborhoods around the
**same labeled support accounts**, keeping the ordinary support-label budget and
all query inputs fixed. That distinguishes sensitivity to account selection
from sensitivity to the graph sampled around those accounts. It is not yet run,
and neither this bound nor that proposed control is an ICLR contribution by itself.

## Reproduction and provenance

Runtime: `b3f0c47a`; independent tensor verifier: `58c97566`. Completed outputs:
`/dataMeR1/phil/gfm/prodigy-mechanisms-role/log/query_exchangeability_full_20260906/`.
The successful smoke is in the separate `query_exchangeability_smoke_20260906/`
directory and is not additional evidence. Large whole-input tensor witnesses
remain on Tucker; only compact tables/receipts are versioned.

Setup modules: `run_query_exchangeability` and `verify_query_exchangeability`.
Run the downstream `analyze_query_exchangeability` module to revalidate all
cohort sums, loss decompositions, bounds, complete grids, input/model hashes and
prior-gradient baseline agreement. Six mathematical unit tests and five
complete-grid/negative-control analysis tests pass; the complete local analysis
suite passes all 78 tests. The namespace-package test suite was loaded by explicit
module names after ordinary directory discovery was not supported.

Worktree: `/Users/philipp/projects/gfm/prodigy-mechanisms`;
branch: `codex/target-performance-mechanisms`. Tucker has its isolated
`prodigy-mechanisms-role` runtime worktree. All computation was CPU-only; no
optimizer updates, target evaluation or production-code changes were made.

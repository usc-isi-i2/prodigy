# Actual training examples: identity conflicts depend on support/query roles

## Result

Hong Kong's actual neighbor-matching episodes contain much more conflicting
supervision for a center-identity/bio-only rule than Ukraine's or COVID's.
But the already completed role-shuffle experiment supplies a counterexample
to the simple explanation that more query–support conflicts imply worse
political transfer. Role shuffling increases those conflicts while improving
political AUC in all three seeds on both episode streams. It simultaneously
reduces conflicting labels among supports. Neither scalar alone has been
isolated as a cause of the learned support-side failure.

This is an **exploratory analysis of complete realized training records**, not
another independently confirmed transfer prediction, a fresh sampler simulation,
or a new training intervention. The prior coverage prediction remains failed.

## Scope and validation

We read all 2,500 consumed batches per model from the nine free-readout controls
(Ukraine/Hong Kong/COVID × three seeds), and all 24 earlier member-policy
controls (Ukraine/Hong Kong × three seeds × four policies). Each batch contains
four episodes of 30 pseudo-classes × (three supports + four queries).

All 82,500 saved batch records pass anchor, ordered-member and member-set hashes;
all 33 complete summaries match the original training verifiers. There are
330,000 recorded episode occurrences, not 330,000 independent observations.
The six older/newer production-policy pairs have exactly matching records,
leaving **27 distinct recorded streams**, not 33 independent exposures. These
records contain center IDs, anchor IDs, roles, and context sizes—not all sampled
context-node identities. Matching records do not establish identical full inputs
between the old member and newer free-readout runs.

Five fixed 500-update windows and first qualifying witnesses per window are
retained. Window reductions, every reported rate, all 240 reused full-model
target cells, and their original checkpoint identities pass a separate local
validation. Six synthetic audit tests and three completed-evidence tests pass.

## An actual Hong Kong training example

In seed 0, update 1, episode 0, merged-graph center **34165817** occurs three times:

| Role | Episode-local pseudo-class | Anchor defining that class | Sampled real nodes, including center |
|---|---:|---:|---:|
| Support | 13 | 34239652 | 67 |
| Query | 24 | 34350986 | 81 |
| Query | 29 | 34298904 | 65 |

Thus the account presented as an example of class 13 must be assigned class 24
in one query occurrence and class 29 in another, in the *same episode*. An
identity-matching rule cannot satisfy all of these assignments. These are
structural pseudo-labels, not semantic account categories or verified annotation
errors. A node can legitimately belong to several anchors' neighborhoods.

The differing context sizes are also direct evidence that these are not three
identical complete graph inputs. The full model can respond differently to them.
This audit does not inspect the account's raw bio or establish which output the
model produced on this training batch. The separate target-case report does
inspect raw text and actual right/wrong predictions: [FINDINGS_EXAMPLES.md](FINDINGS_EXAMPLES.md).

The same phenomenon occurs less often in other sources. Ukraine's first episode
has center 64717 as support for class 14 and query for class 23 (90 versus 85
sampled real nodes). COVID's first episode has center 12803579 as support for
class 2 and query for class 8 (69 versus 55 nodes). Full traceable occurrences,
anchors, roles and record hashes are in `data/consumed_conflicts/records.json`.

## Complete-stream prevalence, not selected anecdotes

Means across three seeds for the standard lowest-ID / sorted-role policy:

| Source | Queries also present as a differently labeled support | Queries sharing their identity with differently labeled queries | Support occurrences sharing their identity with differently labeled supports |
|---|---:|---:|---:|
| COVID | 0.89% | 1.14% | 3.60% |
| Ukraine | 2.10% | 2.95% | 6.80% |
| Hong Kong | 22.22% | 38.53% | 59.29% |

Denominators are role occurrences: 1.2 million queries or 900,000 supports per
stream. A repeated node can contribute to more than one column; these are not
mutually exclusive fractions. Every within-class member is unique, so any
cross-role identity match necessarily points to a different class. The conflict
rate is stable through training: across all 15 Hong Kong seed/time windows,
query–support conflicts range from 22.11% to 22.44%, not merely a startup artifact.

For a deterministic center-only classifier giving equal scores to identical
query centers within an episode, the empirical oracle maximum accuracy is
75.35% for Hong Kong, 98.48% for Ukraine and 99.42% for COVID. The corresponding
minimum cross entropy is .4354, .0216 and .0082 nats. These are optimistic,
label-informed **identity-only bounds, not achieved accuracies or full-model
bounds**. Each repeated query identity occurs once per distinct pseudo-class,
so its best shared prediction gets one occurrence correct; its optimal shared
probability vector is uniform over those classes. Different accounts with
identical bios can only introduce further center-feature ambiguity.

## The existing intervention prevents a simplistic causal story

Hong Kong means across seeds:

| Retention / roles | Queries matching wrong-class supports | Conflicting support occurrences | Identity-only query accuracy upper bound |
|---|---:|---:|---:|
| Lowest IDs / sorted | 22.22% | 59.29% | 75.35% |
| Same lowest IDs / shuffled | 41.15% | 41.19% | 68.48% |
| Uniform endpoints / sorted | 9.79% | 43.21% | 76.36% |
| Uniform endpoints / shuffled | 30.39% | 30.43% | 77.94% |

Shuffling roles within the same retained lowest-ID sets leaves overall repeated
positions unchanged. It worsens the center-only query bound and raises
query–support conflicts by approximately 18.9 percentage points. Nevertheless,
the corresponding existing political target effects are:

| Training seed | AUC change, original / fresh | NLL change, original / fresh |
|---|---:|---:|
| 0 | +.06077 / +.04786 | −.22546 / −.23134 |
| 1 | +.08332 / +.08557 | −.11484 / −.11703 |
| 2 | +.02069 / +.02005 | −.10635 / −.11019 |

These are reused results from the completed member-policy experiment, not six
new replications. The two evaluation streams share each trained checkpoint.
The change also reduces conflicting support occurrences by 18.0–18.2 percentage
points and alters role-specific features/contexts. It therefore does not isolate
either query conflict or support-label ambiguity. The known default-CPU execution
variability further limits attribution of individual effect magnitudes; see
[FINDINGS_NUMERICS.md](FINDINGS_NUMERICS.md).

Conversely, uniform retention with sorted roles substantially reduces
query–support conflicts, but Hong Kong seed 2's political AUC falls on both
streams. All five targets and all three alternative-policy comparisons to the
standard policy remain in the 180-row `data/consumed_conflicts_policy_deltas.csv`.
No policy is selected using these outcomes. This audit does not overturn the
prior failed multi-target coverage prediction.

## What this changes in the research account

The measured pretraining distribution includes role-specific, many-to-many
neighborhood membership—not just distances between bios or whole-graph size.
This gives concrete training examples where a semantic/identity-similarity
heuristic receives competing supervision, much more frequently in Hong Kong.
It is compatible with, but does not explain causally, the independently measured
support-to-label failure pathway and natural-support sensitivity on political
classification.

The next discriminating experiment must separate support-label ambiguity from
query ambiguity while controlling the consumed examples, update numerics, and
training budget. Simply reducing all overlap or adding a graph-similarity score
would repeat an already inadequate test. Any new manipulation must first show
what it changes at the actual training-input/gradient level, retain failures,
and face an untouched target or protocol before an ICLR claim of generality.

Runtime: `62b57f57`, Tucker's dedicated
`/dataMeR1/phil/gfm/prodigy-mechanisms-role` worktree. Source branch/worktree:
`codex/target-performance-mechanisms`,
`/Users/philipp/projects/gfm/prodigy-mechanisms`. No production defaults or
completed checkpoints were changed; the audit used one CPU and no GPU.

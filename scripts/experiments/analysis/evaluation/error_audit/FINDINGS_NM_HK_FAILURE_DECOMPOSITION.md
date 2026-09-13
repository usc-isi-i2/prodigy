# Canonical HK NM failure decomposition

8 September 2026. This joins the complete-input and source-stage caches for all
61,440 HK→HK test query occurrences. It uses no graph loading, encoding, new
sampling, fitting, or GPU work.

## Result

The native HK model is correct on 10,976/61,440 occurrences (17.86%) and wrong
on 50,464. A mutually exclusive descriptive decomposition of those errors is:

| Observed path | Errors | Fraction of errors |
|---|---:|---:|
| Pre-M cosine ranks truth first; final model loses it | 3,135 | 6.21% |
| Pre-M misses; raw full-neighborhood mean ranks truth first | 1,181 | 2.34% |
| Pre-M misses; sampled-node Jaccard ranks truth first | 2,038 | 4.04% |
| Pre-M misses; both raw mean and Jaccard rank truth first | 150 | 0.30% |
| None of these measured summaries ranks truth first | 43,960 | 87.11% |

Raw wins require finite scores for all 30 candidates; Jaccard always has all 30.
The first inspected failure is in the Jaccard-only row: its true class has the
largest sampled-node overlap, while raw and learned cosine and the final model
all choose rivals.

This does not support a general claim that the metagraph causes HK failure.
Against the same pre-M embeddings, the final model rescues 6,139 cosine errors
and loses 3,135 cosine successes, a net gain of 3,004. It is useful overall but
implements a consequential trade: 6.21% of final errors are demonstrably
accessible to this fixed pre-M decision rule.

The raw/Jaccard rows add 3,369 errors (6.68%) where a simple measured input
summary favors truth before the learned common head misses. This is weaker
evidence. Raw neighborhood means discard topology and multimodality. Jaccard
uses global node identity, which independently encoded subgraphs do not expose
explicitly to the model. Together with the pre-M/final losses, the taxonomy
localizes 6,504/50,464 = 12.89% of errors to an observable stage disagreement;
it does **not** causally explain that fraction.

The large residual is stable across episodes: every episode has 66–101 errors
in the residual category (median 85 of 120). Repetition matters to weighting.
Occurrence weighting puts 87.11% of errors in the residual; equal weighting
within each of the 4,987 error query nodes gives 79.14%. Conversely, the
pre-M-correct/final-wrong category rises from 6.21% occurrence-weighted to
14.55% node-weighted. This agrees with the earlier finding that the useful HK
readout gain is concentrated among highly repeated queries.

## Interpretation

What is established is a location map, not a training mechanism. Most canonical
HK errors already lack true-class separation under the learned pre-M mean-cosine
head, and most also lack it under two simple input summaries. That residual can
contain ambiguous episodes, information discarded by the summaries, an encoder
failure, or a failure of the mean-of-three cosine rule. The current evidence
cannot divide it among those explanations.

The strongest next step is to keep these exact HK inputs and compare the native
and foreign checkpoints within the residual, especially native-only successes:
ask which source-trained representation or support-to-class-reference operation
creates HK's final advantage when neither common cosine head succeeds. The
existing source-stage cache already shows the scale: HK's native advantage over
Ukraine is only 2.09 points pre-M but 6.18 points finally, and 4,684/7,565
native-only final successes occur where both common heads fail. Representative
cases from that population should be traced through support values and label
updates before choosing another model change. Weight swapping would mix
coadapted spaces and would not by itself be causal.

The aggregate table is [nm_hk_failure_decomposition.json](data/canonical_split/nm_hk_failure_decomposition.json).
Private occurrence rows are on Tucker at
`/dataMeR1/phil/gfm/error_audit/nm_hk_failure_decomposition_20260908/rows_private.csv`.
Runtime revision `5a3f46f3`, branch `codex/nm-hk-goal-20260908`, worktree
`/dataMeR1/phil/gfm/prodigy-nm-hk-goal-20260908`. The successful join took about
one second after imports on two low-priority CPU threads; no GPU was allocated.

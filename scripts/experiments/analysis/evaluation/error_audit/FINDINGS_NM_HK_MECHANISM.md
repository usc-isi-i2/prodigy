# HK: frozen encoder versus metagraph under support changes

8 September 2026. HK-only runtime completed in **58.10 seconds**, including
standalone graph loading, selected-episode encoding, replay and cache writing
(Python startup/imports excluded). No training or new sampling. The standalone
333,800-node feature artifact reproduces the full original HK input hash exactly.

## Main result: two distinct questions

**The simple prototype head does not fix the original HK failures. But the
native readout loses its advantage when supports are replaced.** This provides
evidence of readout sensitivity under the intervention, without establishing
that the metagraph is the sole cause of ordinary benchmark errors.

The same frozen HK encoder produces every query/support representation. Compare
the native metagraph+decoder with a cosine classifier using each class's mean
support embedding. No head is fitted or selected using these outcomes.

| Input condition | Native accuracy | Prototype accuracy | Decisions |
|---|---:|---:|---:|
| Original supports | **50.0%** | 19.0% | 200 |
| Same supports, fresh sampled contexts | **44.8%** | 20.1% | 1,000 |
| Different supports, fresh contexts | 22.4% | **33.6%** | 1,000 |

Original cases were deliberately selected as 100 native failures and 100 native
successes with degree/frequency-bin matching. The 50% baseline is by design,
not HK benchmark accuracy. Five draws per case are correlated observations.
Only the known true class's support triple is changed; these are diagnostic
interventions, not an operational policy for choosing supports without labels.

## 1. Original errors are not simply “the metagraph ignores a correct prototype”

On the 100 originally failed cases, the prototype succeeds on **1** and fails on
**99**. On the 100 originally correct cases, the prototype retains only **37**
and loses **63**. A second fixed head, mean cosine to the three individually
normalized support embeddings, gets 0/100 original failures and 31/100 original
successes correct.

Thus there is little evidence that replacing the original readout with this
simple classifier would solve those original errors. Failure of a prototype
classifier does not prove the encoder discarded the answer: another decision
rule might extract it, and the input itself can remain ambiguous.

## 2. Under replacement, useful information exists before the native readout

Across the 1,000 alternative-support draws:

| Native result | Prototype result | Draws |
|---|---|---:|
| Wrong | Wrong | 616 |
| Wrong | Correct | **160** |
| Correct | Wrong | **48** |
| Correct | Correct | 176 |

The +112 correct decisions (+11.2 percentage points) are conditional on this
selected intervention sample, not a claimed benchmark gain. The result localizes
some errors to the chosen decision rule: the exact same encoded inputs allow a
simple alternative rule to answer correctly in 160 native-failed draws. The
reverse 48 draws show that the prototype is not uniformly superior.

Originally failed cases reach 12.2% native versus 27.0% prototype accuracy under
replacement; originally correct controls retain 32.6% versus 40.2%. Mean-cosine
accuracy is 32.5% overall, similar to the prototype's 33.6%, without temperature
or head fitting.

A stricter comparison starts with the **37 cases both heads originally got
correct**, giving 185 replacement draws. The native head breaks while the
prototype stays correct in **38** draws; the reverse occurs in **10**. Those
draws involve 15 and 8 distinct cases, respectively. The comparison supports
additional native-readout fragility under replacement while avoiding
a comparison based only on different original correctness states.

Correctness flips across all selected cases are 398/1,000 native versus
258/1,000 prototype under replacement. Under context-only resampling they are
70/1,000 versus 17/1,000, but the prototype's context-condition accuracy is much
lower. Stability alone is not robustness if predictions remain wrong.

## 3. Learned geometry tracks HK loss changes better than raw means

For each draw, compute change in native negative log-probability of the true
anchor. Compare it with change in the true-minus-best-rival margin before the
metagraph. Use the same mean-of-three-cosines scoring rule for raw neighborhood
means and learned subgraph embeddings. Negative correlation means improved
separation accompanies lower native loss.

| Condition | Raw mean-feature margin vs native loss change | Encoded margin vs native loss change | Same eligible draws |
|---|---:|---:|---:|
| Replacement | **+.003** | **−.522** | 421 |
| Context resampling | −.093 | **−.407** | 425 |

These are descriptive Spearman correlations. The shared eligible population
requires nonzero original raw query/all-support summaries and replacement
summaries. With learned embeddings alone, all 1,000 draws are available and
correlations are −.475 / −.460. Within-case correlations across draws also
favor learned geometry: for replacement, median correlation is −.50 encoded
versus −.10 raw across 85 eligible cases. Five draws give coarse, noisy individual
correlations; no independent-draw significance or causal mediation claim.

This explains why the earlier raw-feature diagnostic was insufficient for HK:
it is a weak summary of the space actually used by the learned model. It does
not by itself explain which encoder operations matter or why source training
learned these representations.

Native mean query NLL is 2.264 on the selected original cases, 2.269 under
context-only resampling and 4.153 under replacement. The saved logits also
permit per-case loss/rank analysis. Prototype probabilities reuse the native
learned scale 14.773 without calibration, so cross-head NLL is not presented as
a calibrated likelihood comparison. Top-1 results are scale-independent.

## Interpretation for a fix

### Cached follow-up: are successful draws more similar?

For originally failed HK queries under support replacement, **58/61 successful
draws (95.1%)** have higher learned query-to-support mean cosine than their
original supports; three have lower cosine. Among the **24 queries with both
successful and unsuccessful draws**, successful draws have higher average
learned cosine in **19/24 (79.2%)** and lower in five. Three additional rescued
queries were correct on all five draws and cannot enter that within-query
comparison. The mean/median successful-minus-unsuccessful difference across
the 24 queries is +.06098/+.03477 cosine units.

On the exact same 22 rescued draws with strict raw-feature validity, learned
cosine improves in **21/22**, versus raw neighborhood-mean cosine in **9/22**.
On their 13 eligible mixed-outcome queries, the corresponding within-query
counts are **11/13** versus **9/13**. Thus the contrast is not just different
missing-feature populations. Context-only rescues improve learned cosine in
7/9 draws, and all five mixed-outcome queries have higher mean learned cosine
when successful; these are small counts.

This uses the cached mean-of-three-cosines head, divided by the recorded logit
scale. The query and rival supports are fixed, so differences in its stored
true-minus-best-rival margin equal differences in true-class cosine exactly.
It does not use the native metagraph's logit margin as a cosine surrogate.
No graph loading or inference was needed. No center-only raw feature comparison
was added in this follow-up. These descriptive associations do not establish
that increasing cosine alone causes recovery or explain why training learned
the representation.

[Cached comparison script](compare_hk_rescue_cosines.py) ·
[counts and input hashes](data/canonical_split/nm_hk_rescue_cosines.json).

Paired-support training can target a concrete observed weakness: the native
readout's deterioration under support replacement despite some useful evidence
remaining in the encoder output. The original-error results also caution that
improving readout robustness may not solve the persistent original failures.
Do not remove the metagraph based on this selected sample: it is substantially
better than the simple heads on original supports. A training or full-benchmark
evaluation is still required to demonstrate an actual remedy.

## Exact replay and reusable artifacts

[Aggregate findings and receipt](data/canonical_split/nm_hk_mechanism.json) ·
[aggregation helper](summarize_nm_hk_mechanism.py) ·
[runtime setup](../../../setup/nm_hk_mechanism/README.md).

Private output: `/dataMeR1/phil/gfm/error_audit/nm_hk_mechanism_20260908/`.
`embeddings_and_logits_private.pt` is 73,255,937 bytes and contains all 163
selected complete pre-metagraph episodes (210 embeddings apiece), 6,000 saved
replacement support embeddings, original learned-label vectors and metagraph
edges/roles, case/draw mappings, and all three heads' 30-way logits for the
2,200 selected decisions. Subsequent head/loss analyses need no re-encoding.
The cache and node-level CSV remain private; only aggregates enter the repo.

All original compact-file hashes and the full HK input hash match the canonical
receipt. Standalone feature IDs are mapped by the canonical merged offset
34,148,422; no neighborhoods are resampled or edges reconstructed from a different
graph. Every replacement feature tensor matches the standalone feature rows.
The model loads strictly and its state hash is unchanged. All 200 original and
2,000 intervention correctness outcomes reproduce the saved audit, with maximum
true-probability difference 7.46e-7, below the fixed 1e-5 tolerance.

An independent full-batch forward checks nine selected original episodes:
maximum pre-metagraph difference 7.16e-6 and logit difference 8.59e-6, below
fixed tolerances 1e-5 and 1e-4. Original within-batch learned label indices are
preserved. All nonreplaced pre-metagraph rows remain exactly unchanged during
each intervention. No tolerances or decisions were changed after observing outcomes.

Runtime revision `40d7b5ef`, owned GPU 0, four CPU threads, isolated branch
`codex/nm-complete-input-audit-20260908` in Tucker worktree
`/dataMeR1/phil/gfm/prodigy-nm-complete-input-audit-20260908`. Local runtime
worktree `/private/tmp/prodigy-nm-support-resampling`; findings/helper copies in
`/Users/philipp/projects/gfm/prodigy`, branch `main`. Private Git transport only.
One checkpoint, selected balanced cases, one target and five draws per case;
no training-seed replication or general benchmark improvement is asserted.

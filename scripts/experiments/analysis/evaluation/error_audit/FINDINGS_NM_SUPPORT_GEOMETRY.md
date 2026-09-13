# Fixed-query support changes: does input separation track the flips?

8 September 2026. Completed read-only analysis of all 8,000 saved intervention
predictions (two targets, two models, 200 cases, two conditions, five draws).
There are 4,000 distinct input interventions, shared by the models. Primary
tables below concern native models. No new model forwards or training.

## Takeaway

**Ukraine support replacements often change correctness in the direction of
raw neighborhood-feature separation. Hong Kong does not show the same consistent
relationship.** The earlier cross-episode geometry association is not a universal
explanation of the fixed-query intervention results.

For a fixed query, calculate the true support class's mean-neighborhood cosine
score minus the best rival class's score. Only the true class's supports change;
the query representation and every rival support set remain fixed. Consequently
the margin change is exactly the change in true-class cosine score. This is a
raw-feature diagnostic, not the learned model's logit margin.

## Replacement results

The strict comparison requires nonzero mean-feature summaries for the original
query, all original 90 supports, and the three replacement supports.

| Native target | Rescues with improved margin | Breakages with worsened margin | Eligible replacement draws |
|---|---:|---:|---:|
| Ukraine | **60/70 (85.7%)** | **98/128 (76.6%)** | 665/1,000 (66.5%), 135 cases |
| Hong Kong | **9/22 (40.9%)** | **71/140 (50.7%)** | 421/1,000 (42.1%), 85 cases |

There were 108/61 total rescues and 180/337 total breakages in the original
Ukraine/HK replacement experiments. The table's smaller denominators are the
subset with valid cosine comparisons, not new outcomes. In total 90 Ukraine
and 236 HK correctness flips fall outside the strict geometry subset.
Improved/worsened means a margin change greater than +1e-6 / below −1e-6;
none of these eligible replacement flips are within that numerical tolerance.

For Ukraine, rescued draws improve the margin by +.00877 on average; still-wrong
draws change it by −.00196. Broken correct draws change it by −.00536, versus
+.00103 for retained correct draws. These are outcome-conditioned descriptions,
not separate randomized treatment groups.

For HK, rescued draws have mean change −.00888 (median −.00109), and broken
correct draws −.00484 (median −.00019). Large negative changes affect the means;
the direction counts in the table are more transparent than a mean-only claim.
Only 4/22 eligible HK rescues make the true class strictly closest afterward,
versus 38/70 Ukraine rescues. A rescued prediction does not require winning this
particular raw-feature comparison.

## Within the same query and candidate set

Among cases with both successful and unsuccessful eligible replacement draws,
compare their mean post-replacement margins, weighting each case equally:

| Original cohort | Ukraine: correct draws have higher margin | HK: correct draws have higher margin |
|---|---:|---:|
| Originally failed | 14/17 cases; mean difference +.00817 | 9/13; mean difference −.00289 |
| Originally correct | 14/18; mean difference +.00884 | 10/17; mean difference +.00195 |

HK has mixed signs and sensitivity to large changes. These small case counts
do not support a universal mechanism or independent-draw significance claim.
The query, original candidate anchors and rival supports are fixed within each
case, but support identity, features and topology change jointly.

## Missing-feature sensitivity and exact overlap

As an explicitly weaker sensitivity check, omit rival classes with undefined
zero-vector cosine while requiring finite original/new true-class scores. This
covers 948/1,000 Ukraine and 961/1,000 HK replacement draws. It is not the strict
all-30-class comparison above. Ukraine rescues improve the margin in 88/100
rescue draws (88%), and breakages worsen it in 126/174 (72.4%). HK
figures are 33/60 (55%) and 159/329 (48.3%). The target contrast remains, though
the exact HK fraction depends on the valid population. These sensitivity
comparisons were added after inspecting strict-result coverage.

Exact node-set Jaccard covers all 1,000 replacement draws per target, including
zero features. The query node-ID set is compared with the union of each class's
three support subgraphs, including centers and excluding synthetic pooling nodes.

| Native target | Rescues with improved Jaccard margin | Breakages with worsened Jaccard margin |
|---|---:|---:|
| Ukraine | 54/108 (50.0%) | 101/180 (56.1%) |
| Hong Kong | 20/61 (32.8%) | 198/337 (58.8%) |

For originally failed cases with both correct and wrong replacement draws,
correct draws have larger Jaccard margins in 19/28 Ukraine cases (4 lower,
5 numerically tied), but only 5/24 HK cases (15 lower, 4 tied). Thus increased
shared-node overlap is not a general explanation of support rescues.

## Flips with little change in the summary

Using absolute cosine-margin change ≤.001 as a descriptive scale:

| Native target | Replacement: flips / small-change draws | Same identities, fresh contexts |
|---|---:|---:|
| Ukraine | 23/109 (21.1%) | 16/223 (7.2%) |
| Hong Kong | 26/83 (31.3%) | 7/129 (5.4%) |

The aggregate retains thresholds .0001, .001 and .01; .0001 has very few draws.
These are post-intervention subsets with different case composition, not a
matched comparison establishing an additional causal identity effect. A small
change in one scalar mean-feature margin does **not** mean the actual input or
learned representation barely changed. Topology, feature distributions and
individual support relations can change substantially while this summary stays
similar. Do not call these flips evidence of unexplained model randomness.

The strict HK same-identity subset contains zero rescues: all nine observed HK
context rescues lie outside its valid-feature population. This is missing
coverage, not evidence that context resampling never rescues HK errors.

## Implication for the proposed training experiment

Paired-support training remains a sensible robustness test. Ukraine gives a
specific hypothesis: learning to handle changing support evidence may reduce
errors associated with unfavorable relative neighborhood geometry. HK requires
a broader account than this raw mean-feature score. Do not turn the score into
a universal training target or claim that maximizing overlap fixes NM.

Train on correct answers under multiple support sets and evaluate accuracy plus
support-induced breakages. To localize the remaining mechanism, inspect the
encoder and metagraph representations on these same saved cases; this analysis
has not computed them. No training remedy has yet been demonstrated.

The [completed HK-only encoder/readout follow-up](FINDINGS_NM_HK_MECHANISM.md)
now provides that comparison: the simple prototype head does not solve the
original failures, but outperforms the native readout under support replacement
on these selected cases. Learned geometry tracks native loss changes more
closely than the raw mean-feature summary used here.

## Verification and artifacts

[Aggregate evidence](data/canonical_split/nm_support_geometry.json) ·
[aggregation script](summarize_nm_support_geometry.py) ·
[runtime instructions](../../../setup/nm_support_geometry/README.md).

Private output:
`/dataMeR1/phil/gfm/error_audit/nm_support_geometry_20260908_v2/`.
Every compact input archive matches its recorded SHA-256. Recovered baseline
cosine scores/margins agree with the previous complete-input analysis within
7.16e-7, below the fixed 1e-6 float32 tolerance. Original Jaccard scores agree
within 1e-12. Every saved replacement support ID matches its manifest, and every
saved real-node support feature tensor equals the corresponding backing-graph
rows exactly. All 8,000 predictions join uniquely to their 4,000 shared input
interventions; the original prediction file's hash is recorded.

Runtime revision `2f4df077`, two CPU threads, 17.50 seconds after graph loading.
The initial memory-map loader attempt was unsupported by Tucker's PyTorch;
it produced no results and is excluded. Runtime branch
`codex/nm-complete-input-audit-20260908` in the existing idle isolated worktree
`/dataMeR1/phil/gfm/prodigy-nm-complete-input-audit-20260908`; local runtime
worktree `/private/tmp/prodigy-nm-support-resampling`. Private Git transport only.
Findings and helper copies are in `/Users/philipp/projects/gfm/prodigy`, branch
`main`. No private node-level data are added to git. Five correlated draws per
case, selected balanced cohorts and one training seed per model limit inference.

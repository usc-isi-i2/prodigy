# HK→HK NM: value-path mechanism and a failed bounded repair

8 September 2026. Completed authorized mechanism study and one fixed repair
test. **A local failure pathway is validated; a benchmark improvement is not.**
No GPU, training, new graph sampling, graph loading or encoding was needed.

## What the evidence now supports

The native HK model is better than the foreign Ukraine model on identical
canonical HK inputs, both before the metagraph under a common cosine head
(12.98% versus 10.88%) and at the native decision (17.86% versus 11.68%). On
Ukraine, the corresponding native/foreign scores are 38.89%/17.05% before M and
44.15%/18.15% afterward. Source differences therefore appear at both stages.
These are controlled input comparisons, **not** causal decompositions of training
effects: the encoder and readout were learned together.

HK also has a large shared hard population: both specialists fail on 76.01% of
its original occurrences. Frequent queries, overlapping anchor memberships,
and weak separation under simple raw/encoded geometry are associated with
difficulty. Complete-input analysis already includes sampled context, all 90
supports, all rival classes and the rest of each episode; a query-bio explanation
alone is inadequate. Raw neighborhood feature means and shared-node overlap do
not consistently track HK rescues. Learned cosine tracks them better, but this
correlation does not establish a support-selection policy.

The new controlled result narrows **how changed support evidence affects the
decision**: mainly through projected support-value directions and the resulting
class reference. It does not establish why source training learned that mapping
or why most original HK queries fail.

## Controlled key/value interventions

Reuse the same 200 cases: 100 native failures and 100 correct controls, distinct
query nodes, matched degree/frequency bins, single observed generating-anchor
assignment per query within its episode, and enough alternative supports. These
are selected eligible cases, not a representative error sample. Nearest chooses
three candidates using the known true class; query, rivals and candidate contexts
remain fixed. Seventy-one failures persisted under the earlier random-support
trials.

The actual checkpoint has one M layer and no final-back layer. Query-to-label
messages are excluded; changing support rows leaves final query vectors exactly
unchanged. Swap only the changed supports' projected keys, values, or both.
Keep all queries, weights, normalization statistics and other inputs fixed.

| Intervention | Correct / 100 failures | Retained / 100 controls | Correct / 71 persistent |
|---|---:|---:|---:|
| Original | 0 | 100 | 0 |
| Full nearest supports | 25 | 46 | 5 |
| Keys only | 6 | 66 | 5 |
| Values only | 28 | 48 | 7 |
| New value direction, original magnitude | 26 | 45 | 8 |
| New value magnitude, original direction | 3 | 84 | 1 |

Counts alone obscure the mechanism. **Value-only reproduces all 25 full-nearest
rescues and 51/54 breakages; keys-only reproduces 0/25 and 19/54.** Its six
rescues are different cases. Direction-only reproduces 21/25 and 50/54;
magnitude-only 2/25 and 13/54. Direction-only reproduces all five full-nearest
persistent rescues. Interactions remain; these fractions cannot be added as
independent causal contributions.

Earlier saved-score interventions showed that changing only the true-class score
reproduces all 200 nearest correctness outcomes. Combined with the new tensor
interventions, this localizes the dominant observed pathway to support values
altering that class reference. Attention still matters in other cases. Additive
reference accounting exposes positive/negative messages, self-message, per-edge
output bias, residual and BatchNorm terms; it does not by itself prove a bias or
normalization defect.

## One frozen, label-free repair test

Implement an optional direct geometry path around M. For every query, standardize
each head's scores across its 30 candidates and average native scores with mean
pre-M query-to-support cosine. Equal weights and a 1e-8 standard-deviation floor
were fixed before outcomes. No fitting, weight sweep, query labels or true-class
support selection enters this rule. The feature is NM-only, inference-only,
off by default, and adds no checkpoint parameters. Scores are not calibrated
probabilities.

Primary evaluation uses the **unchanged original canonical** 512 episodes and
61,440 query occurrences per target. Both checkpoints see the same inputs.

| Source → target | Original accuracy | Residual accuracy | Recovered errors | Lost successes |
|---|---:|---:|---:|---:|
| **HK → HK** | **17.86%** | **15.87%** | **3,093** | **4,316** |
| Ukraine → HK | 11.68% | 11.62% | 2,140 | 2,178 |
| HK → Ukraine | 18.15% | 18.93% | 3,959 | 3,478 |
| Ukraine → Ukraine | 44.15% | 42.77% | 2,517 | 3,363 |

**Reject this fixed repair for the primary benchmark.** Native HK loses 1.99
percentage points; 355/512 episodes worsen and 111 improve. Yet equal weighting
per observed HK query node improves 30.71%→31.92%. The 429 nodes occurring 20+
times contribute 43,360 occurrences and 1,373 net lost predictions; the other
frequency groups together gain 150. This is a population tradeoff, not evidence
of memorization or a reason to switch the primary metric after seeing outcomes.

On the selected *nearest-modified* inputs, the same readout improves failures
25→53/100, retains 46→60/100 controls, and improves persistent cases 5→27/71.
It recovers 22 native-missed persistent cases without losing the five native
successes on those modified inputs. But on the selected *original* inputs it
recovers only 1/100 failures and retains only 54/100 controls. Random fixed-bank
draws improve 62→120/500 failures and 170→189/500 controls. Those are five
correlated draws per case, not 500 independent queries or a benchmark gain.

## How much is explained, and what remains open?

- **All original HK errors:** 50,464. No defensible single fraction has an
  established training cause. The original fixed cosine head solves 3,135
  (6.21%) but loses 6,139 native successes. The new residual recovers 6.13% of
  errors while losing 39.32% of successes. These quantify operational recovery,
  not causal coverage or absent information in the remaining representations.
- **Membership accounting:** 3,315 errors (6.57%) choose another test neighbor;
  13,403 (26.56%) choose a train/validation neighbor; 33,746 (66.87%) choose no
  recorded neighbor. These are exact graph facts, not errors relabeled away or
  fractions explained by a training mechanism. Their overlap with readout
  failures prevents summing the percentages.
- **Selected support changes:** the intervention identifies a value-direction
  pathway for most nearest-induced flips in this bank/context. It is not a
  decomposition of all original benchmark errors, and a successful tensor
  hybrid need not correspond to a naturally available support set.

Competing explanations remain: insufficient sampled relational evidence;
information that simple cosine misses; exclusive episode labels for overlapping
neighborhood memberships; source-dependent encoder/readout co-adaptation;
effective member exposure and sorted-ID support/query roles; and normalization
or other reference terms interacting with the learned values. Neither full
graph semantics nor the exact historical training stream has been recovered.
Training already resamples walk endpoints and contexts; ordinary additional
support randomness is not a newly identified remedy.

## Keep the source-training question central

The singleton/pair/LOO and ladder evidence argues against one universal donor
quality or a universal benefit from mixing. Historical 40k matrices establish
asymmetry under their own protocol; their AUCs must not be merged with this
2,500-step canonical accuracy audit. Pair/best-singleton and LOO gaps also
change per-source exposure, so they do not isolate interference.

The recent [three-seed LOO exposure result](../../../setup/nm_loo_schedule_signal/RESULTS.md)
is stronger training-intervention evidence: proportional exposure improves
held-out Ukraine accuracy 33.28%→36.41% at block size one, but redistributes
all-target performance and does not establish a universal exposure optimum.
That connects source allocation to outcomes, not yet to the HK value mechanism.

The most informative next step is to connect **controlled source-training
changes to these same stage and failure measurements**, rather than tune this
failed blend on test labels. First reuse the existing three-seed, two-source
member-retention × role-assignment training factorial, checking its artifact,
budget and split compatibility. Compare its matched checkpoints on the fixed
canonical inputs, retaining both original successes and the frequent-query
population. A change must predictably alter exposure/roles, pre-M separation or
value/reference behavior, and then improve untouched NM accuracy across seeds.
Existing classification results from that factorial do not answer this NM test.

No follow-up run was launched. Rough follow-up estimates: **setup 45–90 minutes**
for compatibility/provenance and replay preparation; **compute 30–60 minutes on
one idle owned GPU** for the full matched checkpoint panel, with uncertainty
from encoder throughput and I/O. Priority jobs must clear first. Cached graph
inputs can be reused; new checkpoint embeddings would still be necessary.

## Evidence, implementation and verification

- [K/V aggregates](data/canonical_split/nm_hk_goal_mechanism.json),
  [direction/magnitude aggregates](data/canonical_split/nm_hk_goal_value_factorial.json),
  [bounded repair aggregates](data/canonical_split/nm_hk_goal_geometry_repair.json).
- [Mechanism aggregation](summarize_nm_hk_goal_mechanism.py) and
  [independent repair checks](summarize_nm_hk_goal_repair.py) retain hashes,
  denominators, recovery/loss counts and numerical differences.
- Prior evidence: [canonical split](FINDINGS_NM_CANONICAL_SPLIT.md),
  [complete inputs](FINDINGS_NM_COMPLETE_INPUTS.md),
  [source stages](FINDINGS_NM_SOURCE_STAGES.md),
  [failure localization](FINDINGS_NM_HK_FAILURE_LOCALIZATION.md),
  [source lattice](../../graphs/transfer_prediction/target_performance_mechanisms/FINDINGS.md),
  [member training controls](../../graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_REPLAY.md).
- All 400 original/full endpoint checks reproduce predictions and correctness;
  maximum true-probability difference is 8.35e-7. Joint K/V reproduces full
  replacement; query outputs, required attention tensors, weights and inputs
  remain unchanged. No validation gate was relaxed.
- Seven tests pass locally and on Tucker, including the production forward
  integration and query-label non-use. Every full-stream native outcome matches
  canonical exports. An independent float64 NumPy implementation reproduces all
  1,800 selected residual predictions. Recomputed geometry differs by at most 6.56e-7: three
  argmax changes across all four cells, including one correctness change for
  Ukraine→HK, and none for native HK. One residual score gap on Ukraine is ≤1e-6;
  none on HK. These are disclosed, not silently substituted into prior findings.
- Three successful CPU passes took **50.18 + 74.40 + 19.26 = 143.83 seconds**,
  using two threads at lowered process priority. Seven-test Tucker execution
  took 0.06 seconds. Setup estimates were reported before each pass. The first
  start failed before inference on a config path and is excluded from results.
  No GPU was allocated; existing jobs and the priority queue were untouched.
- Runtime branch **`codex/nm-hk-goal-20260908`**; local worktree
  `/private/tmp/prodigy-nm-hk-goal`, Tucker worktree
  `/dataMeR1/phil/gfm/prodigy-nm-hk-goal-20260908`. Successful run revisions:
  `af8d8596`, `4dd8e2b0`, `86ceb000`. The experimental model change remains on
  this isolated branch; it is not promoted as a production improvement.
- Private Tucker outputs under `/dataMeR1/phil/gfm/error_audit/`:
  `nm_hk_goal_mechanism_20260908_v2`, `nm_hk_goal_value_factorial_20260908`,
  `nm_hk_goal_geometry_repair_20260908`. No private query rows enter git.
  This report and compact evidence are also placed in the main laptop analysis
  worktree as uncommitted review artifacts; unrelated work is preserved.

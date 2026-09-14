# Compact joint scorer: completed negative result

## Outcome

All six predeclared cells completed. The496,429-scalar joint model loses to its
36,909-scalar structure-only control in every seed on official2018 validation.
Neither arm beats the reproduced AA-DC reference. The frozen advancement rule
fails, and no additional training, tuning, ensemble, or2019 test scoring followed.

| Model | Official2018 Hits@50, mean ± sample SD (%) | Difference vs AA-DC (points) |
| --- | ---: | ---: |
| AA-DC, separately calibrated on2018 | 67.3557 | — |
| Structure-only | 65.2564 ±0.7333 | -2.0993 |
| Joint graph + author vectors + structure | 63.6831 ±0.6934 | -3.6726 |

Joint-minus-control is **-1.5734 percentage points** on average; paired differences
are -3.0407, -0.2846, and -1.3947 points for seeds0/1/2. These are optimization
seeds on one fixed panel, not independent dataset replications. This does not
establish anything about HyperFusion's test result: no test score was computed.

| Seed | Structure selected step | Structure2018 (%) | Joint selected step | Joint2018 (%) | Joint recovered / lost vs AA-DC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 800 | 65.9344 | 250 | 62.8936 | 2879 /5560 |
| 1 | 700 | 64.4781 | 200 | 64.1935 | 3037 /4937 |
| 2 | 950 | 65.3568 | 250 | 63.9621 | 2997 /5036 |

## What the curves show

![Training and historical validation curves](figures/training_curves.png)

The joint model's2017 validation scores peak at200–250 updates, then deteriorate
while sampled training BCE generally falls by orders of magnitude. This is
consistent with overfitting the training panel. Mining makes minibatch loss
nonstationary, so the late loss spikes alone do not establish optimizer divergence.
The structure-only control is much steadier and selects at700–950 updates.
All arms still completed the fixed2000 updates;2018 evaluates the EARLY selected
checkpoints, not the worse final joint checkpoints. Thus the failure cannot be
attributed simply to accidentally evaluating the last checkpoint.

The figure shows raw records every50 updates, no smoothing. The loss is the sampled
training minibatch BCE; the right panels are full2017 selection Hits@50. They are
not2018 assessment or2019 test curves. The historical and official panels differ;
their raw score gap is not a matched within-panel effect.

## Does the joint representation add useful recoveries?

It recovers467/482/593 of the12,185 positives with zero frozen AA-DC base, versus
405/292/355 for the controls. But overall, it gains only24/610/484 additional
AA-DC-missed hits over the controls while losing1851/781/1322 additional existing
AA-DC hits. Every joint seed loses net hits on BOTH repeat and previously unseen
pairs. Richer input did not solve the extreme-negative-tail discrimination problem
under this recipe. A higher count of rescued positives alone is not progress.

The finding rejects this particular graph/author-input bundle, objective, and
training protocol as an improvement over the matched structure-only control. It
does not prove that all compact graph models or the supplied features lack useful
signal. Do not automatically expand the run based on the training curves.

## Contract and admissibility

- Producing revision: `be46110e774938c6b567a5340395ea363535ae24`.
- Branch: `codex/collab-compact-joint`; local worktree:
  `/Users/philipp/projects/gfm/prodigy-collab-joint`; Tucker worktree:
  `/dataMeR1/phil/gfm/prodigy-collab-joint`.
- Runtime: `/dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1` on Tucker.
- Exactly496,385 neural weights plus44 conservatively counted calibration and
  standardization constants for joint;36,865+44 for control. No node-ID table,
  hidden frozen learned module, or ensemble. Supplied128-D benchmark node features
  and graph memory are separately treated as inputs.
- Frozen2015 AA calibration,2016 training,2017 checkpoint selection,2018 assessment.
  Warm-positive historical panels match the previous gate caches exactly, including
  all original pairs, features, and baseline scores. Unique undirected historical
  graphs exclude target-year events. New14-input structural vectors average both
  endpoint orientations; the old comparator remains unchanged.
- Two arms ×three seeds, identical2000-update budget, Adam/BCE/mining policy and
  decoder hidden widths. The isolated random-draw streams match across paired arms;
  the mined identities are intentionally model-specific. This is a representation
  bundle ablation, not an equal-parameter or separate graph-versus-vector ablation.
- All six checkpoint selections froze before2018 feature construction. Earliest
  full2017 Hits@50 maximum selected; no fallback, retry, or sweep expansion.
- The preregistered practical advancement bar required all three joint-control
  differences positive AND joint mean at least1 point above AA-DC. It fails.
- Official2018 panel:60,084 positives and100,000 negatives, unchanged self-pair.
  Pair fingerprint `248062f7572741743e3d6148503d5f743d09bc0e0ca220528aa0440cd4f23762`.
  All metrics use each scorer's recomputed50th-negative threshold and strict `>`.
- Official2018 has been repeatedly inspected in earlier work; this is development
  evidence, not a pristine holdout. Provided node features are not guaranteed
  historical snapshots. The standard audit reads full-split metadata to verify
  identity then discards test arrays; no2019 predictions were produced.

Eight focused tests and eight legacy regression tests passed locally and on Tucker.
The producer audit verifies checkpoint hashes, six-cell completeness, exact selection
rules, cached-input hashes and paired uniform sampling streams. An independent
local NumPy audit verifies the raw score archives, original official pair/feature
fingerprints, all hit/recovery/loss counts, summary statistics, self-pair rule and
advancement decision. No metric or tolerance was changed to admit the outcome.

## Runtime and artifacts

The separate50-update training-only profile used2.46GB peak allocated GPU memory
and0.0275 seconds/update. Production joint cells took about57–59 seconds each and
used2.50GB peak allocated memory. Seeds ran concurrently on GPUs0–2; each session
executed its control then joint arm once and exited. Offline W&B locations are in
the receipts. No production GPU jobs remain from this campaign. Runtime includes
additional preparation and assessment work; the profile is smoke evidence only.

Evidence: [protocol](data/joint_v1/protocol.json), [prepared inputs](data/joint_v1/prepared.json),
[frozen selections and curves](data/joint_v1/selection_frozen.json),
[results](data/joint_v1/results.json), [producer audit](data/joint_v1/audit.json),
[independent score audit](data/joint_v1/independent_score_audit.json),
[profile](data/joint_v1/profile.json).

Reproduce the figure with `plot_curves.py`. Reproduce the independent audit with
`audit_scores.py --runtime <directory containing scores.npz and year2018.npz>
--evidence data/joint_v1 --out <new audit.json>`. Raw archives/checkpoints remain on
Tucker; the local private score-audit copy is `/private/tmp/collab-joint-audit.z0oBTp`.

## Post-hoc failure-case diagnosis

The saved2018 scores show that the failed joint model contains complementary signal,
but uses it too aggressively. Across seeds it recovers2,879--3,037 AA-DC misses while
losing4,937--5,560 AA-DC hits. The consistent subsets make the distinction clearer:
2,170 AA-DC misses are recovered by every joint seed, but4,021 AA-DC hits are lost by
every joint seed. The model is not merely noisy around a few cutoff ties.

The scalar inputs do not cleanly separate the desired and dangerous tail. Of the
consistently recovered positives,95.2% have zero AA,83.8% have a nonzero length-3
signal,78.7% have both endpoints recently active,70.6% have cosine at least0.96,
and66.5% have minimum degree at least10. The16 negatives newly entering every
joint seed's top50 are also all zero-AA:75.0% have nonzero length-3 signal,81.2%
have cosine at least0.96, and68.8% have minimum degree at least10. Mutual recent
activity is the clearest observed difference (43.8% for those promoted negatives),
but this is descriptive evidence from only16 negatives, not an identified cause.

There is still substantial hard positive headroom:15,764 AA-DC misses remain missed
by every joint seed. Their median cosine is0.909 and median minimum degree is5, so a
method that only boosts high-cosine, well-connected pairs cannot close the full gap.

A post-hoc max-score diagnostic preserves AA-DC while admitting sufficiently strong
joint evidence. Scores are normalized at each model's own50th negative, the combined
negative cutoff is recomputed, and a common inspected alpha of0.5 gives68.9463%
mean Hits@50 (69.1299/68.8403/68.8686), versus67.3557% for AA-DC. This is the
largest gain observed in the fixed diagnostic grid, but alpha was inspected on the
repeatedly used2018 panel. It is hypothesis-generating, not a selected validation
result or test estimate. The label-aware union of separate AA-DC and joint hit masks
contains43,349/43,507/43,467 positives (72.15--72.41%); it is not a realizable score
or a formal ceiling.

The evidence supports a narrower next hypothesis: keep AA-DC as an explicit anchor;
learn a conditional correction with a tail-ranking loss that pairs AA-DC-missed
positives against the highest-scoring negatives; penalize loss of high-confidence
AA-DC hits; and add pair/path information that can distinguish equally high-cosine,
high-degree zero-AA pairs. Because the joint model's2017 selection peaked at
77.17--78.73% but transferred to only62.89--64.19% in2018, selection must also test
rolling-year robustness rather than reward one historical panel. Do not rerun the
same direct BCE model with more steps: its selected checkpoints were already early,
and later training degraded2017 ranking.

Evidence: [failure-case receipt](data/joint_v1/failure_cases.json). Reproduce with
`failure_cases.py --runtime <assessment directory> --evidence data/joint_v1
--out <new receipt.json>`. The diagnostic verifies the saved score and panel hashes
and never reads2019 test data.

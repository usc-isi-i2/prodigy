# Temporal path-age prerequisite: insufficient support

The frozen replication prerequisite completed; do not advance to feature training.
This is an inconclusive support failure, not falsification of temporal path age.
No training, GPU use, 2019 array access, or test evaluation occurred.

| Cohort | Year | Usable negative groups | Younger-positive preference | After deleting best 20% groups |
| --- | --- | ---: | ---: | ---: |
| Any seed (primary) | 2017 | 25 | 61.8667% | 52.3333% |
| Any seed (primary) | 2018 | 27 | 62.0988% | 51.2698% |
| Consistent | 2017 | 11 | 52.7273% | 35.0000% |
| Consistent | 2018 | 11 | 58.4848% | 45.0000% |

The primary age direction repeats across years and survives the predeclared trim,
but both years fail the minimum of 30 usable negative groups. The small consistent
cohorts do not reverse in the untrimmed comparison. The descriptive robustness gate
is not a statistical confidence threshold. We did not relax support requirements
or broaden calipers after seeing these outcomes.

Missing-path cases are retained in receipts rather than imputed as age zero. Of
61/46 targeted primary negative groups in 2017/2018, 60/38 find covariate matches,
but only 25/27 have path ages on both sides. Matches use the unchanged all14 and
physical-unit calipers with up to five nearest positives, permitting reuse. The
2017 primary comparison uses 265 positive matches but only 254 distinct positives;
2018 uses 153 matches and149 distinct positives. Pairwise age preference weights
negative groups equally and gives age ties half credit.

The earlier-year models are original2016-trained,2017-standalone-selected models
with warm-only positives. The2018 cohort uses archived fresh2017-trained models
at their original fusion-selected checkpoints. This freezes the cohorts associated
with the original hypothesis; it does not retrospectively choose earlier peaks.
These model and population differences make this directional replication, not a
causal year comparison.2018 also reuses110 of212 current diagnostic cases from the
old path audit:79 positives and31 negatives. It is not independent-data replication.

Independent checks reproduced the gate and trimming from per-group probabilities,
verified the executed source hash, and checked first-observed event dates, exclusion
of prediction-year edges, recurrence handling, and self-pair absence on synthetic
examples. Twenty real cases per year (10 positives,10 negatives) were independently
re-enumerated using sets and first-event dictionaries. Both graphs match frozen
panel edges exactly. Panel,score,standardization,raw-edge,raw-year hashes are in
`data/results.json`; audit and operations records are adjacent.

An initial diagnostic execution had a Git branch/source identity mismatch caused
by overlapping bundle transfers. Its source differed only in the cases selected
for independent auditing. It is rejected as primary evidence. After the job ended,
the owned file/index was restored to the branch revision and the diagnostic was
rerun from a clean source. The frozen protocol and all scientific cohort outputs
are identical; `data/operations.json` preserves the exact deviation and diff.
This is repeated diagnostic execution, not a first-look claim.

Admitted run:7ed4a751; Tucker worktree `/dataMeR1/phil/gfm/prodigy-collab-h3`, branch
`codex/collab-h3`; runtime `/dataMeR1/phil/gfm/ogbl_collab_compact_joint/h3_replication_v2`.
CPU elapsed5.56 seconds, offline W&B run1kz6883n. Private matched examples remain on
Tucker. Prior2019 exploration in the broader research remains disclosed; nothing
here establishes a leaderboard-acceptable win.

Reproduce with `replicate.py --root <runtime-root> --out <new-output>` on Tucker's
prodigy environment; run `audit.py --runtime <new-output> --old <old-path-results>`.

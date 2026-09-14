# Compact validation-frozen specialist fusion

This bounded campaign asks whether the completed structure and joint compact
experts become useful when conservatively fused with AA-DC, following HyperFusion's
specialization insight without using assessment/test score vectors to construct
fusion weights.

The source experts are frozen checkpoints from `compact_joint_v1`: structure-only
(36,865 weights) and joint graph/author/structure (496,385 weights). Together with
shared calibration constants and two selected scalar weights, the full three-expert
predictor counts 533,296 learned/fitted scalars. No model is retrained and no node-ID
embedding is added.

For each expert and seed, the 2017 50th-negative score is frozen as its scale. One
common alpha tuple per fusion rule is selected by mean 2017 Hits@50 across all three
seeds. Ties prefer the smaller total expert weight, then lexicographic order. The
three predeclared rules are AA+structure, AA+joint, and AA+both. The fusion score is
the maximum of normalized AA-DC and whichever normalized expert scores are enabled.
All scale constants and alphas remain unchanged on official 2018 validation.

Primary estimand: AA+both minus frozen-2015-calibrated AA-DC on 2018. Advancement
requires at least +1 point in the three-seed mean and the both-expert rule beating
both single-expert controls in every seed. Failure stops the branch; it does not
trigger a wider grid. Official 2018 has already been inspected in prior work, so
this remains post-hoc development evidence. There is no 2019 test scorer.

Run in a dedicated Tucker worktree on an owned GPU:

```bash
python scripts/experiments/setup/ogbl_collab_compact_fusion/run.py select \
  --out /dataMeR1/phil/gfm/ogbl_collab_compact_fusion/fusion_v1 --device cuda:0
python scripts/experiments/setup/ogbl_collab_compact_fusion/run.py assess \
  --out /dataMeR1/phil/gfm/ogbl_collab_compact_fusion/fusion_v1 --device cuda:0
python scripts/experiments/setup/ogbl_collab_compact_fusion/run.py audit \
  --out /dataMeR1/phil/gfm/ogbl_collab_compact_fusion/fusion_v1
```

Run `--dry-run` before selection. Keep the source runtime at
`/dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1` unchanged. Preserve small
results and receipts under the matching analysis directory; checkpoints and raw
score arrays remain private runtime artifacts.

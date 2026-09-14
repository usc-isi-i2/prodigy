# Fresh-negative joint two-model ensemble

Authorized September 14, 2026. No training. Reuse candidate_fresh_v1's three
2017-trained, 2018-fusion-selected checkpoint score archives; no new checkpoint
selection. Compare pairs (0,1), (0,2), (1,2) using equal raw-logit averages in
float64. Highest 2018 Hits@50 wins, ties lexicographically. Standalone control is
highest 2018 standalone at the same existing checkpoints, ties lowest seed.
Require strictly greater validation Hits before any new test scoring. Freeze
selection.json before the separate test command. No test-based extension.

Count two full model states plus conservatively duplicated original auxiliary
state and the averaging coefficient: 992,897 scalars, below 1M. No third model
is deployed. Checkpoint tensor counts are checked, and existing 63-scalar
auxiliary budgets retained conservatively even though AA mixing is removed.

Use Tucker's prodigy environment, CPU only, offline W&B. From repository root:

```sh
python scripts/experiments/setup/ogbl_collab_fresh_ensemble/run.py select --source /dataMeR1/phil/gfm/ogbl_collab_compact_joint/candidate_fresh_v1 --validation-panel /dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1/assessment/year2018.npz --out /dataMeR1/phil/gfm/ogbl_collab_compact_joint/fresh_ensemble_v1 --dry-run
```

Remove --dry-run to select. Only after gate_passed=true, repeat with phase test
and --test-panel /dataMeR1/phil/gfm/ogbl_collab_compact_joint/official_test_fusion_v1/year2019.npz.
Both phases reject output overwrite. Source scores and checkpoints must match
producer hashes; validation scores must reproduce recorded checkpoint metrics.
Official OGB evaluator and independent strict-50th-negative computation must agree.

All six validation cells are required. Missing or mismatched artifacts abort.
A failed validation gate completes the experiment without test access. Otherwise,
one frozen pair is evaluated on 2019; failure to exceed 0.7129 ends this branch.
No SD is inferred from overlapping pairs. Prior repeated test exploration,
negative-exposed training and test-supervised diagnostics remain disclosed.
A numerical win alone does not establish leaderboard acceptance.

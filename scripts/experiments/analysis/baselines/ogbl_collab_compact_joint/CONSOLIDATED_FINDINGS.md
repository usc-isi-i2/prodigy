# Consolidated ogbl-collab findings: sub-million HyperFusion challenge

Closeout (2026-09-14): the final user requirement is a **leaderboard-accepted**
result, not merely a larger test-informed number. Earlier test-informed proposals
below are historical diagnostics, not authorization for submission or further runs.
Acceptance has not been established. All proposed next experiments are parked;
see [retirement record](../../../../../cleanup_records/2026-09-14-collab.md).

## Objective and current answer

The explicit objective is to show that a model with fewer than one million learned
or fitted inference scalars can beat HyperFusion's reported ogbl-collab test
Hits@50 of **0.7129 ± 0.0018**. We have not yet established that claim.

The strongest defensible forward result is the fresh-negative compact candidate at
**0.697979 ± 0.001786** test Hits@50 with **496,448** inference scalars. Its best
standalone joint seed is **0.701569**, leaving **0.011331** to HyperFusion. A
test-supervised cross-fitted tree reaches 0.849864 below the parameter cap, showing
within-panel fitting capacity, not generalization. It is not benchmark-valid and
does not transfer from 2018 to 2019.

## Evidence ladder

| Result | Hits@50 | Size | Evidence status | Decision |
| --- | ---: | ---: | --- | --- |
| Official AA-DC, 2018 validation | 0.673557 | 0 | reproduced baseline | Anchor/control |
| Official AA-DC, 2019 test | 0.680243 | 0 | reproduced baseline | Anchor/control |
| Original compact joint, 2018 validation | 0.636831 ± 0.006934 | 496,429 | audited development evidence | Reject original BCE recipe |
| Original compact structure, 2018 validation | 0.652564 | 36,909 | audited development evidence | Better than joint, below AA |
| Validation-frozen AA + joint, 2018 | 0.682433 ± 0.007134 | 496,430 | audited post-hoc forward-year diagnostic | Positive but unstable |
| Validation-frozen AA + both experts, 2018 | 0.677779 ± 0.005306 | 533,296 | audited post-hoc diagnostic | Reject global two-expert fusion |
| Test-tuned global fusion, 2019 | 0.694288 ± 0.000563 | 496,448 | test-informed development only | Still below target |
| Exact released HyperFusion rule, percentile interface | 0.662127 ± 0.003545 | 533,296 | test-informed protocol diagnostic | Rule hurts weak bases |
| Best test-tuned rank sum | 0.69155 | 533,296 | test oracle diagnostic | Global fusion ceiling is too low |
| Label-aware union | 0.73052 | 533,296 | non-realizable oracle | Pair complementarity exists |
| Cross-fitted test-supervised shallow tree | 0.849864 ± 0.005310 | ≤543,696 | reproducible but non-admissible test supervision | Learnability only |
| Same tree, trained 2018 and frozen for 2019 | 0.677035 ± 0.005160 | ≤535,376 | forward-year post-hoc evidence | Reject current gate |
| 2018 training with reused official negatives | 0.825394 ± 0.003345 | 496,448 | invalid for generalization: 99,984/100,000 test negatives exposed | Do not cite as a clean win |
| Fresh-negative validation-select/refit candidate | **0.697979 ± 0.001786** | **496,448** | strongest audited forward result; prior test exploration disclosed | Current base for next step |

The reported dispersions are sample SD across three optimization seeds where three
seeds exist. They are not episode-sampling confidence intervals.

## What HyperFusion actually does

The released Collab script stacks predictions from AGDN, E2N/E2E-GCN, and PLNLP.
It computes pairwise cosine distances separately for validation positives,
validation negatives, test positives, and test negatives. Distances between zero
and 0.1 populate a model-by-partition incidence matrix `H`; the propagation matrix
is `H @ H.T`, and its row sums weight the final model-score sum.

Consequently, labeled test-positive and test-negative score partitions affect the
reported test fusion. This makes the released result test-informed. It does not by
itself prove the leaderboard number is wrong, and we lack their full base score and
configuration artifacts, so we cannot attribute how much of 0.7129 comes from strong
bases versus fusion.

Applying the exact rule to our AA, structure, and joint scores fails because the
neural bases are weak on 2019. Native-score fusion gives 0.634743 mean. A monotonic
empirical-percentile interface gives 0.662127. Even a weight grid optimized directly
on the test labels peaks near 0.692. HyperFusion-style global mixing therefore
cannot rescue the original experts.

Sources: [released HyperFusion Collab code](https://github.com/zhangxwww/HyperFusion/blob/master/HyperFusion_collab.py)
and [OGB leaderboard](https://ogb.stanford.edu/docs/leader_linkprop/).

## Confusion/error-accounting conclusions

The original joint scorer has complementary positives but damages AA's high-value
tail. Across seeds on 2018 it recovers **2,879--3,037** AA misses while losing
**4,937--5,560** AA hits. The intersection contains 2,170 AA misses recovered by
all seeds, 4,021 AA hits lost by all seeds, and 15,764 positives missed by every
joint seed. Increasing recall or adding a positive bias is therefore not the fix;
it also promotes the hardest negatives and displaces correct AA positives.

Among consistent recoveries, zero AA, nonzero L3, mutual recent activity, high
feature cosine, and reasonable minimum degree are common. The negatives promoted
by all seeds share most of these properties. Mutual recent activity differs most,
but the persistent promoted-negative sample is only 16 and cannot support a robust
rule. A matched all-input path audit removes the apparent raw path-count advantage.
Younger path-formation age remains a modest descriptive lead, not a validated feature.

## Why successive approaches failed

1. **Feature-only MLP and hard-negative variants:** weak test ranking and insufficient
   discrimination at the 50th-negative boundary.
2. **Original joint GNN:** learns useful rescues but applies them too aggressively;
   selected 2017 checkpoints transfer poorly to 2018.
3. **Global max/weighted fusion:** improves AA modestly but one global scale cannot
   survive the year shift. The 2017 ranking of fusion rules reverses in 2018.
4. **Released HyperFusion construction:** assigns weights from whole score-vector
   similarity, not pair-specific correctness; weak experts drag down AA.
5. **Pair-conditional tree gate:** fits within-year labels extraordinarily well,
   but 0.8499 cross-fitted test performance collapses to 0.6770 when trained on 2018
   and applied forward to 2019. It learned year-specific boundaries.
6. **Training on 2018 official negatives:** reaches 0.8254, but nearly every official
   test negative was reused in training, invalidating a generalization claim.

## The fresh-negative repair

The concurrent `candidate_fresh_v1` campaign repairs the negative-exposure defect.
It trains on 2017 with deterministic fresh negatives, selects checkpoint, AA base,
alpha, and normalization using 2018, then reinitializes and trains on 2018 fresh
negatives for exactly the selected update count. It evaluates 2019 once. Both fresh
negative panels have zero overlap with official validation and test negatives.

| Seed | Selected updates | Frozen alpha | Fusion test | Standalone joint test | Recovered/lost vs AA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 150 | 1.50 | 0.696281 | 0.700360 | 1,519 / 776 |
| 1 | 200 | 0.75 | 0.699842 | 0.693065 | 1,844 / 936 |
| 2 | 250 | 1.00 | 0.697813 | **0.701569** | 1,458 / 644 |

This is the largest credible progress toward the target. Fusion is worse than the
standalone joint expert for seeds 0 and 2, reinforcing that AA mixing is no longer
the main opportunity. The selected steps are early, consistent with rapid overfit.

## Current decision and next bounded test

Stop work on the original global fusion and shallow gate families. The next cheapest
test is a two-model ensemble of the freshly refit joint experts:

- two 496,385-weight joint models plus shared small calibration state remain below
  one million total inference scalars;
- choose the seed pair and combination rule only from the 2017-to-2018 selection
  score archives;
- apply that frozen pair once to the existing 2018-to-2019 refit score archives;
- do no new training and no test-dependent pair selection;
- require test Hits@50 greater than 0.7129; otherwise stop seed ensembling.

This test is warranted because the current best standalone model is 0.701569 and the
three seeds make different errors. It is not guaranteed to recover the remaining
1.13 points. If it fails, progress requires a stronger single expert trained for
rolling forward transfer—multiple historical transitions and a top-tail ranking
objective that protects strong AA decisions—not another fusion grid.

Because 2019 has already been repeatedly inspected, even a validation-selected win
must be described as post-hoc development unless accepted through an external fresh
evaluation or benchmark-maintainer-approved protocol.

## Provenance map

### Canonical detailed findings

- Compact joint history, negative-overlap audit, fresh-negative candidate, failure
  cases, and path diagnosis: [FINDINGS.md](FINDINGS.md).
- HyperFusion rule, test oracle, cross-fitted gate, and forward gate:
  `scripts/experiments/analysis/baselines/ogbl_collab_hyperfusion_oracle/FINDINGS.md`
  on branch `codex/collab-hyperfusion-oracle`.
- AA/MLP complementarity:
  `scripts/experiments/analysis/baselines/ogbl_collab_complementarity/FINDINGS.md`
  on branch `codex/collab-validation-complementarity`.

### Branches and relevant commits

- `codex/collab-compact-joint`: fresh candidate producer `bcb77879`, independent
  audit `07a416c2`, consolidated closeout `e40c3790`.
- `codex/collab-compact-fusion`: frozen-fusion result `440677cc`.
- `codex/collab-hyperfusion-oracle`: exact rule `9dae3711`, gate result `4ad979db`,
  forward failure `ee0f4a73`.
- Original compact checkpoint revision: `be46110e774938c6b567a5340395ea363535ae24`.

### Tucker runtime roots

- Original compact experts: `/dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1`.
- Fresh-negative candidate: `/dataMeR1/phil/gfm/ogbl_collab_compact_joint/candidate_fresh_v1`.
- Frozen compact fusion: `/dataMeR1/phil/gfm/ogbl_collab_compact_fusion/fusion_v2`.
- HyperFusion oracle scores: `/dataMeR1/phil/gfm/ogbl_collab_hyperfusion_oracle/oracle_v3`.
- Cross-fitted gate and replay: `/dataMeR1/phil/gfm/ogbl_collab_hyperfusion_oracle/gate_v1`
  and `gate_v1_repeat`.
- Forward gate: `/dataMeR1/phil/gfm/ogbl_collab_hyperfusion_oracle/forward_v1`.

Failed setup-only roots `oracle_v1` and `oracle_v2` contain no result: the first hit
a validation-specific constant guard during shifted test construction; the second
used the wrong detached Python environment. `oracle_v3` is the completed run.

### Operational note

Earlier runaway GPU jobs were terminated during this investigation. Subsequent
campaigns used dedicated Tucker worktrees and named tmux sessions. Completed sessions
exited; unrelated concurrent jobs and GPUs 4--7 were left untouched. Before any new
launch, re-inventory current processes, tmux sessions, worktrees, and GPUs 0--3.

## One-sentence takeaway

We have shown that a sub-million model has enough within-year signal to dominate the
reported score, but the signal does not yet transfer across years; the best defensible
forward system is 69.80%, and the immediate remaining bet is a validation-selected
two-model fresh-expert ensemble under the same sub-million cap.

# Temporal residual gate pilot on ogbl-collab

## Constrained rescue (constrained_v1; frozen before launch)

Run with `--warm-only --constrained-rescue --out
/dataMeR1/phil/gfm/ogbl_collab_temporal_gate/constrained_v1`.
Only nonself pairs with zero raw AA and zero prior pair events may receive a
learned correction. All other scores remain the frozen 2015-calibrated base in
the existing log-score coordinates. This does not preserve the separately
2018-calibrated reference score, nor guarantee preserved hits when cutoffs move.
Eligible corrections remain signed, allowing hard-negative suppression.

Train on eligible 2016 positives and negatives only; retain the same 2048-positive,
1024-uniform-negative, 1024-hard-negative batch sizes. Mine the top 2048 eligible
negatives every ten steps. Nonself eligibility applies equally to both labels.
Keep full warm_v1 panels for calibration, feature standardization, 2017 selection,
and official 2018 assessment. No changes to graphs, features, negative arrays,
481-parameter architecture, three seeds, 400 steps, optimizer, strength grid,
tie-breaking, or full-panel Hits@50 selection. This tests the combined eligibility
constraint and subgroup training, not separate causal effects of those changes.
Checkpoint metadata records the new scoring flag; previous checkpoints must not
be interpreted as constrained scorers merely because shapes match.

Report all seeds against warm_v1, frozen AA-DC, and official-calibrated AA-DC,
including recoveries, losses, repeat/new slices, protected-score equality, and
negative-cutoff movement. Success requires consistent gains, not picking a seed.
No sweep expansion or test scoring; stop after the fixed run even if it fails.
This remains post-hoc 2018 development. CPU eight threads with offline W&B;
warm_v1 took 78 seconds. Use idle dedicated gate worktree and tmux
`collab-temporal-gate-constrained`; evidence goes in `data/constrained_v1/`.

## Warm-endpoint follow-up (warm_v1)

Run with `--warm-only --out /dataMeR1/phil/gfm/ogbl_collab_temporal_gate/warm_v1`.
The sole intervention is requiring both positive endpoints to have at least one
strictly earlier graph edge in 2015 calibration, 2016 training, and 2017 selection.
Graphs remain unchanged. Generate historical negatives BEFORE filtering positives,
excluding all target-year positives, so the negative arrays exactly match pilot_v1.
All architecture, optimizer, seed, budget, checkpoint and strength-selection rules
below remain fixed. Feature standardization is recomputed on the filtered training
panel as prescribed by the existing algorithm. Calibration is also recomputed.
Official 2018 assessment remains unchanged; no test scoring. The dataset loader's
existing full-split identity audit still reads test metadata, not model scores.
This is post-hoc development, not a pristine validation holdout.

Primary outcome: three-seed mean 2018 Hits@50 versus original pilot and official
AA-DC, with per-seed recovered/lost positives versus both AA baselines. Require all
three seeds, matching negative arrays and unchanged assessment pairs/features.
Stop after this fixed run regardless of outcome. CPU eight threads, offline W&B;
previous full pilot took 86.52 seconds on Tucker. Use dedicated tmux
`collab-temporal-gate-warm` in the existing idle gate worktree. Preserve original
pilot evidence; write new evidence under analysis `data/warm_v1/`.

Objective: test a small, temporally transferable correction to AA-DC as a possible
route toward HyperFusion's reported 71.29% test Hits@50. This pilot does not open
test or claim that an improvement on validation beats that test score.

## Frozen pilot contract

- **2015:** calibrate upstream AA-DC gate/anchor on pairs from 2015 and graph events
  strictly before 2015. Freeze calibration and the score normalization constant.
- **2016:** train a residual MLP over pair attributes. Its graph contains
  only events before 2016. Historical negatives: 100,000 deterministic uniform
  unordered pairs, deduplicated, excluding positives from that target year only.
- **2017:** select residual checkpoint and strength; graph events before 2017.
- **2018:** official validation panel, scored after selection is written. Historical
  AA-DC calibration stays frozen. Also report the already-reproduced AA-DC calibrated
  on 2018 as a separately labeled, stronger reference, and verify that replay.
- **2019 test:** never scored; read only by the existing dataset fingerprint audit.

Author features are the supplied static OGB inputs, as in prior baselines. They
are not guaranteed to be historical snapshots. We do not reuse MLP checkpoints
trained through 2017 for earlier-year training examples. Edge-weight lookup for
each graph excludes later-year arcs before constructing the pair dictionary.

Features: log AA, log L3 (computed for AA-zero pairs, matching upstream), external
ratio, minimum/maximum endpoint LCC, minimum/maximum log degree, raw feature
cosine, log prior pair event count, capped time since last event, minimum/maximum
log recent endpoint event counts, and AA-zero indicator. All attributes are
computed without target-year positive edges in the graph. Pair identity determines
the zero-self rule for every scorer, independently of labels.
Endpoint min/max and cosine features are symmetric. Upstream AA-DC/L3 scoring
preserves supplied endpoint order for reproduction; no new symmetry claim is made
for the upstream L3 feature or baseline.

Score: `log(max(AA_DC, 1e-6) / cutoff_2015) + strength * residual(features)`.
The residual is `12*tanh(MLP(features)/12)` with one 32-unit ReLU hidden layer,
zero-initialized final layer, and features standardized on 2016 only. The base
score is frozen. A negative residual can suppress hard negatives; a positive
residual can rescue positives. Self-pairs receive score -1e9.

Three optimization seeds (0,1,2), 400 steps, Adam lr 0.001, weight decay 0.0001,
gradient clipping 5. Each step samples 2,048 positives, 1,024 uniform negatives,
and 1,024 from the current top 2,048 scored 2016 negatives (refreshed every ten
steps). Hard-negative mining never uses 2017 or 2018 labels/scores. BCE loss;
selection every 20 steps, strengths [0, 0.25, 0.5, 1]. Earliest step then smallest
strength breaks ties. The zero-strength baseline is an eligible selection.

All three selected models must finish. Preserve selection manifests, checkpoint
hashes, feature/pair fingerprints, temporal boundary checks, curves and full score
archives. W&B offline; machine-readable files are authoritative. Historical
negative sampling and static node features differ from prospective deployment,
so this remains a benchmark pilot. The official validation panel has been seen in
earlier experiments; it is not a pristine held-out discovery set.

Stop after the fixed pilot; no test, parameter sweep expansion or retraining on
2018. Report historical selection and 2018 outcomes even if the baseline wins.

```bash
python scripts/experiments/setup/ogbl_collab_temporal_gate/run.py \
  --out /dataMeR1/phil/gfm/ogbl_collab_temporal_gate/pilot_v1 --dry-run
python scripts/experiments/setup/ogbl_collab_temporal_gate/run.py \
  --out /dataMeR1/phil/gfm/ogbl_collab_temporal_gate/pilot_v1 --threads 8
```

This bounded pilot uses CPU pair features and a sub-1,000-parameter model. Measure
per-year feature time and per-model training time in the runtime record. Run in
the dedicated `prodigy-collab-gate` Tucker worktree and `collab-temporal-gate` tmux.

# Compact joint graph/feature/structure scorer

## Frozen question and decision

Can full author vectors plus end-to-end graph context improve extreme-tail ranking
over an otherwise matched structure-only decoder, within 1M learned scalars?
This is a bounded step toward the user's explicit HyperFusion/<1M benchmark goal,
not a claim that this architecture can win or a commitment to a manuscript.

Two arms, optimization seeds 0/1/2, six required cells:

- `joint`: mean SAGE 128->256->256->256, ReLU after the first two layers;
  pair input `[z_u*z_v, abs(z_u-z_v), symmetric14]`; decoder 526->256->128->1.
  Exactly 496,385 neural parameters, no node-ID embedding table.
- `structure`: the same decoder hidden widths/activations, with only symmetric14
  inputs (14->256->128->1): 36,865 neural parameters.

All numerical AA calibration entries plus the 28 fitted standardization scalars
are conservatively added to each model's inference count in `prepared.json` and
`selection.json`. Provided benchmark node features and graph storage are inputs,
not hidden trainable embeddings. Report their runtime/memory separately. Models are
reported independently, never ensembled. Every total must remain below 1,000,000.

This is a representation-bundle ablation, NOT an equal-parameter comparison and NOT
a separate identification of author-vector versus graph-aggregation contributions.
The two arms share panels, structural features, optimizer, sampling policy, uniform
draw stream, step budget, checkpoint rule, and decoder hidden widths. Mined negative
identities differ because each model ranks its own fixed training-negative pool.

Primary estimand: joint-minus-structure official 2018 validation Hits@50, reported
as all three paired seed differences and their mean. Also compare both with the
separately 2018-calibrated AA-DC reference, report recovered/lost positive hits at
each scorer's own recomputed 50th-negative cutoff, new/repeat net hits, and zero-base
recoveries. A positive pilot requires all three joint-minus-control differences
positive and the joint mean at least1 percentage point above AA-DC. The1-point bar
is a predeclared practical threshold, not a statistical confidence threshold.
Smaller gains are incremental, not grounds for automatic expansion. This is a go/no-go for further investigation,
not a significance claim or a HyperFusion win. Do not advance on one lucky seed.
If the condition fails, report that this bundle failed under the fixed budget; do
not automatically expand it. Even a positive pilot does not authorize test scoring.

## Inputs, chronology, and objective

Reuse the temporal gate's audited warm-endpoint panels and frozen 2015 AA calibration:
2016 trains; 2017 selects; official 2018 assesses only after all six selections freeze.
Historical graphs contain only earlier-year events; each GNN graph is unique,
unweighted, undirected. Target-year positive edges are absent. Historical negative
arrays are generated excluding all target-year positives BEFORE warm-positive
filtering. Historical negatives can include previously observed pairs. Official
2018 pairs remain unchanged, including its negative self-pair.

The 14 features are the previous 13 inputs plus log1p(frozen AA-DC). Compute the
transformed vector for BOTH endpoint orientations and average them, including
the log-transformed base feature. This makes input semantics symmetric; the old
AA-DC comparator and all old features/scores remain unchanged. Forward calibration
is frozen in 2015, not recalibrated for the learned symmetric features. Fit mean/std
on the full 2016 pair panel only. Static supplied 128-D node features are used as-is;
they are NOT guaranteed to be historical feature snapshots.

Balanced BCE; Adam lr0.001, weight decay0.0001, clip norm5. Each update has2048
positive,1024 uniform negative,1024 mined negative draws with replacement. Mine
the top2048 of the100k fixed training negatives every10 updates. A dedicated seeded
generator isolates draws from model initialization; record/verify uniform-stream
hashes across arms. Self-pairs receive -1e9 independently of labels.

Exactly2000 updates per cell; full2017 Hits@50 every50. Earliest maximum wins,
without a baseline fallback. No early stopping, extension, architecture search,
ensemble selection, automatic retry, or automatic test evaluation. A last-step
selection is budget-limited evidence, not convergence. This budget is deliberately
larger than the400-update tiny-gate studies, so those are contextual, unmatched
references. Official2018 has been repeatedly inspected in prior work; it is
development evidence, not an untouched holdout. The standard dataset audit reads
test split metadata for fingerprint verification then discards test; no test scorer
exists in this experiment.

## Phases and reproduction

`run.py --dry-run` emits the full machine-readable contract and expected grid.
Use the same Git revision for preparation, training, assessment, and audit; runtime
hashes and code revision are checked before reuse. Existing output directories
cause a hard failure. There is no silent resume; completed cells are retained, and
failed-cell recovery requires explicit inspection and a documented new output path.

Dedicated worktree: `/dataMeR1/phil/gfm/prodigy-collab-joint`, branch
`codex/collab-compact-joint`. Runtime root:
`/dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1`.

1. `python scripts/experiments/setup/ogbl_collab_compact_joint/run.py prepare --out <root>`
   freezes the contract and writes only2015-2017 panels plus node features.
2. `... run.py profile --out <root> --device cuda:0` runs50 training-only updates,
   offline W&B, no validation or assessment. Separate `profile/` artifacts are smoke
   only. Measure step time and GPU memory before choosing safe launch concurrency.
3. `bash scripts/experiments/setup/ogbl_collab_compact_joint/launch_tucker.sh <gpu> <seed>`
   explicitly runs structure then joint for one seed, once, with no retry loop.
   Each seed gets one dedicated tmux session; only GPUs0-3 are permitted. Inspect live
   processes before launching. User authorized trying the bounded experiment.
4. After all six `selection.json` files pass checks, run `... run.py assess --out
   <root> --device cuda:0`. It freezes the complete selection manifest BEFORE
   constructing any2018 features/scores. No automatic assessment supervisor is used.
5. `... run.py audit --out <root>` recomputes metrics and checks completeness,
   checkpoints, provenance, earliest-best selection, and matched uniform streams.

Use the documented Tucker prodigy environment and8 CPU threads per process. Every
substantive phase logs offline W&B and authoritative local JSON/checkpoints. Preserve
small admitted evidence in `scripts/experiments/analysis/baselines/ogbl_collab_compact_joint/data/`;
large pair arrays, checkpoints and node features remain private runtime artifacts.

Tests: `python -m unittest discover -s scripts/experiments/setup/ogbl_collab_compact_joint
-p 'test_*.py'`, plus the temporal-gate regression tests. The shared PRODIGY model,
episode pipeline, old checkpoint schemas and default gate path are unchanged.

## Bounded frozen-fusion follow-up

`fusion.py --runtime <joint_v1> --out <new directory> --device cuda:0` replays
the three selected joint checkpoints on 2017 (requiring exact Hits parity), then
freezes a common max-fusion alpha and each seed's normalization before reading
saved 2018 predictions. `--dry-run` prints the full contract without artifact IO.
No training or official 2019 test access occurs. AA uses the SAME 2015 calibration
in both years; the separately 2018-calibrated AA score is a contextual reference.
This is a new development diagnostic, not a pristine holdout or an automatic
extension of the failed original advancement rule. All three seeds are required;
there is no ensemble, retry, or tuning after assessment. A gain must survive all
three seeds to motivate another decision; no test scoring is authorized.

The two-base HyperFusion-style diagnostic applies the released cosine-distance
threshold0.1 and H H-transpose aggregation to these SAME transformed scores, with
2017 partitions only versus adding 2018 positive/negative partitions. It is an
adaptation of the fusion rule, not a reproduction of HyperFusion's three-model
submission or raw score basis. For two bases, weights are necessarily equal or
zero, so this cannot generally identify a useful adaptive preference. Zero weights
receive no fallback. This limitation is predeclared, not repaired after inspection.
The primary test remains frozen max fusion. All scalars across components are
conservatively bounded by496,434. Runtime uses offline W&B and saves protocol,
selection, replay scores, and results. Reproduce tests with `test_fusion.py`.

## Explicitly authorized official-test tuning

The user's subsequent instruction prioritizes beating the reported71.29% number
and explicitly authorizes official2019 scoring and test-informed mixing selection.
This supersedes the no-test restriction ONLY for this new follow-up; previous
records remain unchanged. It is test-tuned development, not untouched-test evidence.

`official_test.py --runtime <joint_v1> --out <new directory> --device cuda:0`
scores the same three selected joint checkpoints without retraining or ensembling.
Use `--dry-run` first for the complete bounded grid. Max fusion tests20 fixed
weights and two AA baselines (2015-frozen and reproduced2018-calibrated), using
2019 negative cutoffs for normalization. Report common-weight mean/SD, all seed
scores, single best seed/weight, standalone joint, and AA controls. All40 grid
cells ×3 seeds are required. Stop after this grid; no automatic extension.

Test graph contains train+2018 validation events only. GNN adjacency is unique,
unweighted and undirected. AA weights exactly follow upstream's2018 decay anchor
and unit validation weights; no2019 event is added. Repeated pairs legitimately
observed before2019 remain. Structural feature age is evaluated at2019, keeping
the original2015-calibrated14 inputs and2016 standardization. Reproduce the
official AA test score68.0243% as an admission check. AA reference calibration
uses2018, not test positives. Model/state hashes and full split fingerprint are
checked; all metrics must match OGB. Every learned scalar is counted, including
the additional reference calibration, normalizers and mixing coefficient:
at most496,448. No node-ID table, hidden learned model, or multi-seed ensemble.

Runtime: `/dataMeR1/phil/gfm/ogbl_collab_compact_joint/official_test_fusion_v1`.
Dedicated existing worktree/branch as above, dedicated tmux session; GPUs0–3 only.
Offline W&B and local protocol/source/panel/score/result records are mandatory.

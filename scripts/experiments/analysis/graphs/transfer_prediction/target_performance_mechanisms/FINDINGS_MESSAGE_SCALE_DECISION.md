# Message-scale decision: support direction, not a pure norm effect

**SUPERSEDED DECISION, 7 September 2026:** the original data and interpretation
below are preserved as a dated snapshot. The approved final-contrast and 50k
tests are now complete; see `FINDINGS_CLASS_REFERENCE_DECISION.md`. The 50k
political direction-only failure narrows the earlier explanation, so this
title must not be reused as a cross-configuration conclusion. References below
to the 50k test as "not launched" describe the earlier decision, not current
run status.

**PUBLICATION STATUS.** Read-only `gh repo view` verified
`usc-isi-i2/prodigy` is PUBLIC. Earlier commit `abe57b4e` pushed per-episode
support-geometry summaries and experiment provenance before this was checked.
No bios, model checkpoints or private per-query tensors were uploaded in these
commits. The user subsequently accepted continuing with the public repository
and plans to make development private later; no deletion or history rewrite
was requested. This completion follow-up remains local-only. Neither the first
snapshot nor the full completion analysis was pushed. Private datasets,
credentials and restricted artifacts remain outside public Git transport.

## Completion update: all 900 cells

The original run completed all 60 model/target/stream blocks, 900 conditions,
in 2,615 seconds. No restart, new experiment, or runtime revision change was
needed. The complete result is in `data/message_scale_20260907`; the separate
first 540-cell snapshot is preserved. All four metrics in those 540 cells are
exactly unchanged in the final export.

Validation passes: 27,900 stage/cohort geometry rows, 57,600 exact query checks,
zero historical endpoint logit error, zero unassignable radial norms, unchanged
inputs/weights/operators, and finite metrics in every cell. The analysis unit
test passes. Only aggregate metrics, geometry and provenance were retrieved;
private per-query predictions remain on Tucker.

### What the remaining targets decide

Mean AUC changes in points, three initialization seeds x two episode streams:

| Source / target | Message removal | Pre-meta direction only | Pre-meta norm only | Multiplier 0.01 |
|---|---:|---:|---:|---:|
| Hong Kong / twibot20 | -8.851 | -8.569 | +0.805 | -5.767 |
| Ukraine / twibot20 | -5.353 | -5.532 | +0.137 | -4.069 |
| Hong Kong / ukraine-suspended | +2.383 | +2.118 | -0.075 | +2.311 |
| Ukraine / ukraine-suspended | +1.266 | +1.096 | +0.073 | +1.013 |

Bot direction-only and removal hurt in all six comparisons for each source.
The pre-pooling direction swap agrees: -8.694 points for Hong Kong and -5.312
for Ukraine; its norm-only counterparts are -0.231 and +0.023. Direction thus
carries both the CP/FB rescue and the BOT damage. This strengthens localization
but contradicts any claim that directional movement is intrinsically harmful.

For Hong Kong on BOT, direction-only also reduces accuracy by 6.483 points and
F1 by 7.949 points, while **improving NLL by 0.145** (all six comparisons).
Removing context is therefore not a universal performance repair, even within
a single target. Lower NLL alone does not demonstrate better calibration.

Suspension is not literally flat: Hong Kong removal moves mean AUC from .5033
to .5271 and Ukraine from .4993 to .5119. Both have mixed signs across the six
comparisons and remain near chance; do not present this as a third useful
repair target. Election AUC changes are small/mixed (Hong Kong removal +0.379,
Ukraine +0.032 points). All target results remain available in the full table.

The original explanation survives with a sharper boundary: paired direction
swaps reproduce target-dependent changes in ranking; a pure radial explanation
does not. They do not establish a common unhelpful direction, isolate the
specific affine/nonlinear cause, or predict the sign before target outcomes.
Nor does holding queries fixed establish that query context always helps;
the historical role evidence retains its target/seed-dependent qualifications.

The single next test remains the source-validation-selected Hong Kong 50k
checkpoint below, with its own actual protocol. **It has not been launched.**
No manuscript expansion, reviewer package, or new public upload was made.

The updated one-page figure adds the BOT norm/direction comparison, explicitly
labels full completion, and includes the NLL counterexample. The PDF skill's
render-and-inspect workflow was followed; the initial page, figure and builder
are archived under `archive/first_540_cells/` beside the standalone deliverable.
Completed PDF SHA256:
`11a695026a0b423db96c377250eb427eae255b35afebc565058b6b9ef627a6f9`.
The main manuscript retains SHA256
`b2f2ff5ad58f1b89a7f088baf31bf022a594325dc79305524f62165d020cbe9f`.
The completion automation `finish-message-scale-decision` was successfully
set to **PAUSED** after delivery; the broader publication goal was not changed.
Local branch remains `codex/role-topology-interactions`, worktree
`/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`, HEAD `abe57b4e`.
No new commit or push was made in this completion follow-up.

## Preserved first-decision interpretation (540-cell snapshot)

7 September 2026. This is a bounded decision note, not a manuscript expansion.
The original full scale run continues, unchanged. Its complete political,
election and page blocks are analyzed here: **540 cells**, all six original
checkpoints, both streams, all 15 conditions and all 32 batches. Bot and
suspension scale blocks are not claimed complete in this snapshot.

## What the tests decide

Mean AUC changes in points, three initialization seeds x two episode streams:

| Hong Kong support intervention | covid-political | facebook-page-reference |
|---|---:|---:|
| Zero projected messages | +7.716 | +1.224 |
| Zero-message directions, intact norms, before metagraph | +7.477 | +1.104 |
| Intact directions, zero-message norms, before metagraph | +0.575 | +0.040 |
| Direction-only swap before pooling | +8.044 | +1.174 |
| Norm-only swap before pooling | -0.545 | +0.181 |
| Message multiplier 0.01 | +6.592 | +1.071 |

Direction-only gains have the same positive sign in all six comparisons on
both repair targets. These paired swaps support direction as the principal
carrier of the repair; they are not an additive causal decomposition or a
proof that every conceivable norm intervention is ineffective. Zero-message
scaling retains message-MLP biases, self terms and saved BatchNorm. The small
nonzero multiplier does not preserve intact behavior, and 1e-6 approaches zero.

Pure vector-size reweighting and a discontinuous message-presence explanation
are contradicted. A learned non-radial response is supported; the exact origin
among affine terms/nonlinear operating range is not isolated. Bypassing support
background BN alone does not reproduce the rescue (HK mean -0.765 CP, -3.525 FB
points), and is not equivalent to training a model without BN. At political
message receivers, intact-to-zero cosine is .956 before BN and .888 after BN;
about 15% of coordinate signs change. Roughly 83% of political support-node
occurrences receive a message; only about 4% receive more than one. Nodes with
no incoming message remain exactly unchanged at this stage.

## What does not generalize yet

The existing nine-source role replay was complete, contrary to the review's
scope objection. Suppression helps in both streams for 5/8 foreign CP sources
and 3/8 foreign FB sources; all nine sources lose bot AUC. These are descriptive
signs, not significance claims, and use distinct historical seed-0 checkpoints.

Saved embeddings and logits were analyzed without any new model forwards.
The political-trained source has almost HK's unit-support dispersion (.237 vs
.242), but suppression loses 5.74 points rather than gaining 9.92. The
low-dispersion suspension-trained source (.045) gains 7.35. Dispersion therefore
fails as a general warning sign. HK output total-variation sensitivity is
similar on CP (.219) and BOT (.209) despite opposite AUC changes. Class-mean
cosine associates with CP gains outside HK (Spearman .79 on stream means), but
not consistently across targets. All these relationships are exploratory;
outcomes had already been inspected. None is a validated intervention selector.

## One next generality test - nominated, not launched

Use the existing source-validation-selected step-50,000 Hong Kong model from
`cp_hk_twitter_nm_bio_02_07_2026_08_58_57`. The saved best model and that numbered
checkpoint have identical model digest
`c392cb264a05ac9a21dd6d3afe4590d3a9e7e8f74079da3e2a89fe35668a6a21`.
Preserve its original S/U/M, 768-dimensional input, one-hop/3-shot protocol;
do not silently instantiate the recent final-core configuration. Test the
predeclared CP/FB/BOT role and norm/direction pattern, with native/raw/untrained
readouts, rather than select a favorable checkpoint or target after inspection.
This addresses brief training, not public-benchmark or mean-trained generality.

Contemporaneous requirements files for all six current training arms record
PyG 2.3.1. Together with their recorded training source revision, this supports
sum aggregation during training. This is reconstructed provenance, not a new
training replay. Exact paths and the long-run configuration are in
`data/message_scale_training_provenance.json`, including the versioned
[PyG implementation](https://pytorch-geometric.readthedocs.io/en/2.3.1/_modules/torch_geometric/nn/conv/message_passing.html).

## Deliverable and continuation

One-page PDF (main manuscript unchanged):
`/Users/philipp/projects/gfm/paper/transfer-prediction/message-scale-decision-2026-09-07/output/pdf/support_direction_decision.pdf`.
Initial PDF SHA256:
`61bd39e3b14bb09cb66975570f3499b8437e5323e5e1f52c33d7011b8081bff8`.
The builder and figure are beside it. Initial page is visually verified.

Local branch `codex/role-topology-interactions`, worktree
`/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`.
Running Tucker worktree `/dataMeR1/phil/gfm/prodigy-message-scale`, pinned at
`14d909e89f07c7adc16b2dc9962333fb7198dbc4`; tmux
`message_scale_full_20260907`; output `log/message_scale_full_20260907`.
CPU only, four threads. **Do not pull/switch this checkout or relaunch the run.**
The existing snapshot passes 34,560 exact query checks, zero historical endpoint
logit error and zero unassignable radial norms. Full grid completion is false.

The separate idle Tucker analysis worktree is
`/dataMeR1/phil/gfm/prodigy-scale-analysis`. No training/evaluation ran there:
only 90 saved-embedding and 90 saved-logit analyses. Their outputs and receipts
are under `data/cached_support_geometry_20260907` locally.

Reproduce the first snapshot with `analyze_message_scale --input
<analysis>/data/message_scale_first_decision_20260907/raw --complete-targets
covid_political election2020 facebook_page_reference --output <new analysis>`.
It validates whole target blocks, not partial seeds/streams. No DONE marker is
fabricated for the ongoing full job.

Completion follow-up `finish-message-scale-decision` checks this existing run
every ten minutes, with a finite September 7 deadline. After real full DONE:

1. Retrieve protocol, metrics, receipts, DONE and all ten geometry JSON files,
   not private per-query tensors, into a **new** full-results raw folder.
2. Run the same analysis module without `--complete-targets`, output
   `data/message_scale_20260907`; require 900 cells and all validation gates.
3. Use `plot_message_scale_argument` with its default three swap targets to
   include BOT, and inspect all five targets before refining the conclusion.
4. Follow the PDF skill; archive the first decision page, update only that
   standalone one-page argument/status, render and visually verify it. Preserve
   this first 540-cell data snapshot and keep it distinct from the full result.
5. Report the completed result and pause the heartbeat. **No additional
   experiment, main-manuscript expansion, reviewer release or publication-goal
   completion is authorized by this finite follow-up.**

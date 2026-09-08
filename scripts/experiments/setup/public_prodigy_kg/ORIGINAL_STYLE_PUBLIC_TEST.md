# One original-style public PRODIGY test

Protocol fixed 2026-09-07 before this run's target results. This directory
contains execution helpers, not a new model implementation. Use the clean
original repository at commit
`107ba57234d3188227cda5b78a2dbcfb84a1c694`, obtained through Git from
https://github.com/snap-stanford/prodigy. Do not run the social-model fork in
its place. No manuscript expansion depends on an unfinished run.

## Assets and isolation

- Tucker data: `/dataMeR1/phil/data/prodigy_public_original`.
- Original code: `/dataMeR1/phil/gfm/prodigy-public-upstream`, detached at the
  revision above. Never update it while a run is live.
- Staging helper worktree: `/dataMeR1/phil/gfm/prodigy-publickg-assets`,
  detached at `5ac1dfcd`; session `publickg-assets`.
- Physical GPU 2 is nominated for the subsequent model run; use only 2/3.
- Use the existing `prodigy` environment; no package replacement is prescribed.
  W&B stays offline; canonical params, metrics, checkpoints and receipts stay
  in the experiment's own output directory.

Tucker uses Python 3.11.15 with Torch 2.0.1+cu118, PyG 2.3.1,
SentenceTransformers 2.2.2 and Transformers 4.29.2. Python 3.11 removed
`random.sample(set)` used by upstream `MulticlassTask.sample`. A narrowly
scoped compatibility wrapper converts sets to tuples in their existing order,
as Python 3.10 did, using the same RNG. It neither sorts nor changes the task
population, and is recorded separately from observation-only logging. No
aggregation, label-mask, model, or objective correction is bundled with it.

`stage_assets.py` is dry-run by default; `--execute` downloads the official
Wiki/FB15K-237 ZIPs, validates pinned HTTP size/ETag, records SHA256, checks
ZIP paths and CRCs, and extracts into a new directory before final rename.
No model/data pickle is loaded during staging. Archives are retained.
Download is 14,940,330,324 bytes; full archive expansion is 29,815,382,619
bytes. These are distribution sizes, not RAM or GPU estimates.

No license grant was found in the inspected repository/resource index or
archives. Public accessibility is not a redistribution permission. Do not
publish the archives, derived examples, checkpoints, or paper artifacts;
reviewer redistribution remains TBD.

## Fixed training contract

Use `pretrain_cmds['PRODIGY']` from upstream `kg_commands.py`:

- Wiki, genuine `cls_nm` mixture (classification:NM = 98:2).
- `S2,UX,M2`, positive-only metagraph edges, 256 hidden dimensions.
- Cached MPNet node/relation features; 768 inputs plus two head/tail flags.
- UX concatenates head, tail and global-max features before projection.
- 15 ways, 3 supports and 4 queries per way, batch 10, 7 loader workers.
- AdamW, learning rate/weight decay .001, no dropout, ND.5/NZ.5 augmentation,
  auxiliary attribute weight 1,000; native 2-hop Wiki neighborhoods.
- Full released run: 10,010 updates. Seed 0 is added explicitly.
- **Nominate `state_dict_8000.ckpt` before target evaluation.** In the released
  zero-indexed loop it contains **8,001 updates**. It is not the terminal
  checkpoint. The paper's reported step-8,000 evaluation motivates this fixed
  rung, not a checkpoint filename specified by the released eval command.
  Do not substitute the highest checkpoint or one selected on FB.

A separate three-update smoke may test execution with zero workers and one
validation/test batch. It is not a trained result, must have a different
output directory, and is never resumed as the full run.

## Fixed target contract and released-code discrepancies

FB15K-237 relation-type classification, 20 ways, 3 supports/way,
4 queries/way, batch 1, 500 tasks, 15 loader workers, label IDs 0--199,
`ignore_label_embeddings=True`, no target augmentation or tuning.

Preserve these executable choices and record them in output metadata:

1. Initial `eval_only` evaluation leaves the model in **training mode**.
   Gradients/weight updates are disabled, but BatchNorm uses current-batch
   statistics and updates buffers. The primary result is not frozen-BN
   inference. Do not choose a mode after seeing accuracy.
2. The upstream `kg_task_no_labels_split` aliases a label array while toggling
   masks. With the released uncapped support pool, `split='test'` actually
   uses original train+test edges (~90%) for supports and original valid
   edges (~10%) for queries. They are disjoint; it is not the intended 70/20
   split. Preserve it for this literal released-code test. A loader-only
   preflight verifies the actual task's support/query candidate pools, and
   all primary readouts share saved complete native inputs. This does not
   independently audit unique original edge IDs in every captured episode:
   the original subgraphs retain center pairs, not unique edge IDs.
3. PyG 2.3.1's instantiated `aggr_module` executes **sum**, although the SAGE
   layer's later `aggr` string says mean. Record both. Changing the operator
   would create a different trained model; no such correction is part of
   this run.

The paper describes 30 training ways, 3 evaluation queries, and a capped
10-example support pool; the released commands differ. Its 72.04% FB accuracy
is context, not an exact-parity threshold or the source of checkpoint selection.
This test cannot establish that a correctly implemented mean-trained model
has the same mechanism.

## Ordered decision, not checkpoint search

First establish useful native-model performance on the nominated checkpoint
and fixed episodes. Compare native accuracy with predetermined raw-feature
and untrained-encoder cosine prototypes, using the same supports/queries:

- Raw pair descriptor: concatenate the cached head and tail feature vectors
  in order; no relation label embedding or held-out target-edge feature.
- Untrained descriptor: the actual S2+UX output of the original architecture,
  strictly loaded from this full training run's saved `initial_model.ckpt`
  (not the smoke run or a newly reseeded model), before either M layer,
  retaining the same released training-mode BatchNorm behavior.
- L2-normalize example descriptors, average the three supports per class,
  normalize each prototype, and use cosine argmax; no hyperparameter tuning.

Only an improvement over both readouts with paired uncertainty supports the
claim that this test addresses a weak-model explanation. If a faithful
matched readout cannot be implemented, report the missing comparison rather
than declaring that objection closed. A miss is retained, not replaced by
another checkpoint or task.

Then the single role-conflict prediction is the conjunction:

`accuracy(support-only suppression) > accuracy(intact)` and
`accuracy(query-only suppression) < accuracy(intact)`.

Suppress only background message-passing edges before S2; preserve nodes,
features, head/tail roles, pooling and metagraph edges, labels, and episodes.
Use intact/support-only/query-only as the primary arms. Cache the complete
native sampled model inputs for all 500 episodes before model mutation,
including nodes, edges, features, head/tail/role flags and labels; all readouts
and interventions consume those exact tensors. With 15 workers, matching a
seed alone is insufficient evidence of pairing. Restore identical checkpoint
buffers and RNG state at the start of each paired stream. Record
query-state drift: with M2 and batch-statistics normalization, fixed queries
are not an invariant. Do not retrofit the single-M localization test.

For the same three role streams, save CPU-cloned S2+UX point vectors and the
final M2 point/class vectors, local row/role metadata, and native logits with
per-file hashes. These are observation-only captures from the already planned
forwards, not additional arms or decision criteria. Pre-metagraph query drift
can arise from shared current-batch normalization; it must not be confused with
query changes propagated through class references in M2. Final-state vectors
support inspection of that distinction without rerunning models. They remain
private with the native input captures, never in public Git.

Report paired episode accuracy differences with a fixed-seed, 10,000-draw
episode bootstrap (seed 20260907), using two-sided 95% percentile intervals,
plus effect sizes and all primary arms. Both native-minus-prototype lower
bounds must exceed zero. The support-minus-native lower bound must exceed
zero and the query-minus-native upper bound must be below zero. All conditions
are required; non-excluding intervals are inconclusive, not positive evidence.
These intervals concern the sampled task distribution conditional on this
graph/checkpoint; shared entities preclude a claim of independent graphs or
training seeds. Do not substitute pooled binary AUC for native accuracy.

If quality or role-conflict evidence fails, stop this nominated public test
and retain its outcome. If both hold, the supported claim is role-dependent
end-to-end context utility in a useful public original-style model. It is
**not** independent implementation generality, fixed-query mediation, a
cross-task validation of the political K/V mechanism, or a deployable selector.

## Reproduction commands

Use the wrapper's dry run first. It validates the pinned upstream path and
prints the native command/protocol; `--execute` requires completed asset
receipts and a fresh output directory. It observes original training/eval
without replacing layers, objectives, samplers, or normalization.

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
python scripts/experiments/setup/public_prodigy_kg/run_native.py \
  --phase smoke --upstream /dataMeR1/phil/gfm/prodigy-public-upstream \
  --root /dataMeR1/phil/data/prodigy_public_original \
  --output /dataMeR1/phil/gfm/prodigy-publickg-native/log/publickg_smoke_20260907 \
  --gpu 2
```

Only after a successful smoke, use `--phase train` and a distinct
`publickg_train_20260907` output. Launch long commands in a dedicated tmux
session with conda's PATH export inside the session command. Do not append
`--execute` until ready to execute; a dry run is not a completed experiment.

## Execution record (local/private; not paper results)

- Both archives completed download, CRC validation and extraction. Direct
  graph-tensor reads verified Wiki: 4,838,243 nodes, 5,856,692 edges, 639
  relations; FB15K-237: 14,543 nodes, 268,039 edges, 200 relations.
- Wrapper/tests and public catalog metadata are on branch
  `codex/role-topology-interactions`, code commit `4c14de12`; local worktree
  `/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`.
- Nineteen tests passed locally. Nineteen were run on Tucker, with the optional
  laptop-path AST integration test skipped there; the other eighteen passed.
- Native worktree `/dataMeR1/phil/gfm/prodigy-publickg-native` is detached at
  `4c14de12`. It invokes the separate untouched upstream checkout.
- Three-update smoke `log/publickg_smoke_20260907` completed successfully.
  The original model has 2,894,723 parameters before freezing its label table.
  Actual SAGE modules report `aggr='mean'` but runtime class `SumAggregation`;
  the observed initial source evaluations use training-mode BatchNorm.
- Full training was launched separately as `publickg-train` on physical GPU 2,
  output `log/publickg_train_20260907`, console
  `log/publickg_train_20260907.console.log`. No automatic target eval or
  checkpoint-selection sweep is attached. Check the live tmux/process and
  output status before deciding whether it is still running.
- Cached readout/intervention helpers were reviewed and transported in
  code-only commit `e3448fe8`. Their isolated Tucker worktree is
  `/dataMeR1/phil/gfm/prodigy-publickg-eval`, detached at that revision;
  the live trainer remains at `4c14de12` and upstream at its original pin.
- All 60 implementation tests passed locally. On Tucker, 59 passed and the
  optional laptop-path recipe check was skipped; the actual upstream
  synthetic CPU forward/replay test passed there. These are implementation
  checks, not public-benchmark performance.
- The one-episode real loader fixture completed at
  `prodigy-publickg-evalprep/log/publickg_layout_20260907`: 140 subgraphs,
  60 supports/80 queries, 13,372 nodes, 14,695 edges. The candidate pools
  contain 241,235 support / 26,804 query edge indices with zero overlap.
  This does not imply disjoint entities, center pairs, or neighborhoods.
  The cached helpers validated the 140-by-1,536 raw descriptor layout and
  removed exactly 6,012 support edges or 8,683 query edges while preserving
  pooling/metagraph tensors. Zero model forwards or performance calculations
  were used for this fixture check.
- This is execution progress, not a public-benchmark result. No target
  performance has been inspected, and no manuscript or one-page PDF changed.
- At 2026-09-07 01:28 PDT, the live training process (PID 4118917) had saved
  `state_dict_1000.ckpt` and was running its periodic Wiki evaluations.
  The nominated 8,000 checkpoint was not yet available. All three pending
  commands below passed dry-run planning on Tucker without creating outputs.
- The subsequent observation-only update `8995a4cc` adds the planned role
  state captures described above. It was pushed on the same code branch and
  installed in the idle `prodigy-publickg-eval` worktree, now detached at
  that revision. All 61 tests passed locally; on Tucker, 60 passed and the
  optional laptop-path check was skipped. The actual upstream synthetic
  forward test verifies saved-state logit reconstruction and unchanged
  tensors/weights. At 02:21 PDT the original training PID was live at 4,313
  updates; its worktree and upstream source pins remain unchanged. No public
  target evaluation has run, and the decision gates are unchanged.

## Pending evaluation sequence

**Completed role result, 2026-09-07 03:43 PDT:** all 500 episodes per arm and
all 1,500 private state captures completed. Native accuracy 0.73795;
support-only suppression 0.39925; query-only suppression 0.38995.
Support-minus-native difference -0.3387, paired95% CI
[-0.345500625, -0.331925]; query-minus-native -0.348,
CI [-0.35475, -0.3414]. The role-conflict conjunction **fails**: both roles
benefit from context in this public setting. All native replay classes/labels
match; the original logit tolerance verdict remains false as disclosed.
Do not replace the checkpoint or task, add a favorable public condition, or
present this as a replication of harmful support processing. Advisor decision:
stop this experiment block and integrate this strong boundary into the
one-page argument. No new model forwards are nominated.

Role execution update, 2026-09-07 03:39 PDT: the competing mechanism process
exited and GPU 3 was verified idle. Clean evaluation/upstream pins remained
`30df8cb5` / `107ba572`. The nominated cached-input role test launched in tmux
`publickg-roles-accuracy`, output `log/publickg_roles_accuracy_20260907`, with
the passing amended quality run and explicit accuracy-equivalence policy.
No competing-run outcomes were substituted or used to change these arms.
Intact/support/query predictions and observed states are pending.

Quality result, 2026-09-07 03:35 PDT: amended quality completed with exact
native predicted classes/labels across 500 episodes; original logit verdict
remains false. Native accuracy 0.73795, raw 0.724125, untrained 0.698025.
Native-minus-raw difference 0.013825, paired 95% CI [0.007424375, 0.02025];
native-minus-untrained 0.039925, CI [0.0336, 0.046225]. Both original scientific
quality criteria pass. The next nominated output is
`log/publickg_roles_accuracy_20260907`, using quality run
`log/publickg_quality_accuracy_20260907` and explicit
`--replay-policy accuracy-equivalence-20260907`.

Role launch was deferred: a separate `publickg-mechanism` tmux session took
GPU 3 (Python PID 4187456) from worktree `prodigy-publickg-paired-state`, running
`run_mechanism ... --parity-atol 0.0001`. It is not this fixed cached-input test.
Do not stop that process, duplicate it, or silently substitute its outputs.
Check resource availability and coordinate before launching our role test.

### Engineering amendment decided before baseline/intervention outcomes

Implementation update: code-only commit `30df8cb5` implements explicit
`--replay-policy accuracy-equivalence-20260907`; original strict behavior is
still the default and the original `replay.passed` logit verdict remains
unchanged in output. All 62 local tests passed; Tucker passed 61 with one
optional laptop-path check skipped. The idle evaluation worktree was updated
through Git, then the amended quality run launched in tmux
`publickg-quality-accuracy`, fresh output
`log/publickg_quality_accuracy_20260907`. Original failed quality artifacts
remain intact. Roles must explicitly request the matching amended policy and
use this amended quality output only after its scientific gate passes.

2026-09-07, after the six-forward native-only repeatability diagnostic:
the original elementwise logit gate remains **failed**, not retrospectively
passed. Repeated native forwards on fixed episodes 0 and 4, with identical
checkpoint, buffers, inputs and restored RNG, yield non-bit-exact values
already at the first SAGE output. On episode 4, repeat 1 versus repeat 0 and
repeat 2 versus repeat 0 fail the original tolerance themselves (logit maximum
absolute difference 5.7220458984375e-06). Every diagnostic prediction and label
matches. Thus unchanged execution exhibits numerical variability; this does
not isolate a particular CUDA/scatter operation as its cause.

Advisor decision, independently assessed before any raw/untrained or role
outcomes: continue in **separately named amended outputs**, requiring exact
query labels and predicted classes for all 500 native replays, finite logits
and matching shapes/dtypes. Preserve the original tolerance values and all
per-episode logit diagnostics, including their failed status. Do not tune a
larger epsilon. This is accuracy-equivalence verification, not bit-exact
representation or mediation verification. The checkpoint, all 500 inputs,
initialization, model/BN mode, baseline definitions, three role arms, and
bootstrap scientific acceptance conditions remain unchanged. Do not launch
roles until amended replay and both original quality comparisons pass.

The diagnostic code is commit `50b110c9` on
`codex/role-topology-interactions`, local `.worktrees/role-topology`, installed
through Git in the idle `prodigy-publickg-eval` worktree. Private evidence is
`log/publickg_replay_diagnostic_20260907/{protocol,diagnostic,execution_status}.json`.
The six diagnostic forwards completed without weight changes. At this record,
the amendment is a documented decision, **not yet implemented or executed**;
baseline and role results remain unobserved. Original failure outputs stay intact.

Execution update, 2026-09-07 03:26 PDT: native evaluation completed all 500
episodes (40,000 queries), accuracy 0.73795. The paired quality runner, tmux
`publickg-quality`, Python PID 4176776, subsequently completed its 500 native
replays and exited with a failed replay gate. All 500 query-label arrays and
all 40,000 predicted classes match the saved native outputs, but 61 episodes
fail the fixed elementwise logit tolerance (439 pass). Maximum absolute logit
error is 1.33514404296875e-05; maximum tolerance ratio is 3.129341926904086.
Weights were unchanged. Raw/untrained comparisons and role interventions did
not run. This is a numerical replay failure, not a failed model-quality or
role-conflict hypothesis test; its numerical cause has not been established.
Keep the failure artifacts and tolerance unchanged. Do not treat the native
accuracy alone as resolving the weak-model objection. Saved diagnostics are in
`prodigy-publickg-eval/log/publickg_quality_20260907/native_replay.json` and
`execution_status.json`. No revised scientific protocol has been executed.

Execution update, 2026-09-07 03:21 PDT: the nominated checkpoint was saved at
03:19, SHA256 `5e48d5abc0660f7e537d79d6ccd74150aed38eecc8ca90a048adb6dc94aafee7`.
The native 500-episode evaluation is now running in tmux `publickg-native-eval`,
Python PID 4174329, from clean detached helper revision `8995a4cc`, against
the unchanged original upstream pin. Physical GPU 3 was verified idle before
launch; training PID 4118917 remains untouched on GPU 2. The first shell launch
exited before Python because the fresh helper worktree lacked its `log/` parent;
creating that directory allowed the unchanged command to start. This was not a
model restart or a protocol change. Console output is at
`prodigy-publickg-eval/log/publickg_native_eval_20260907.console.log`.
Native evaluation completion, cached quality/replay, and conditional roles
are still pending; no public result is claimed by this launch record.

Run from the isolated `prodigy-publickg-eval` worktree. These are dry runs by
default; append `--execute` only once the nominated checkpoint exists and
physical GPU 3 is idle. The native evaluation must finish before quality;
quality and replay must both pass before roles. No automatic watcher is
attached to the training process.

```bash
python -B scripts/experiments/setup/public_prodigy_kg/run_native.py \
  --phase eval --upstream /dataMeR1/phil/gfm/prodigy-public-upstream \
  --root /dataMeR1/phil/data/prodigy_public_original \
  --checkpoint /dataMeR1/phil/gfm/prodigy-publickg-native/log/publickg_train_20260907/state/Wiki_PT_PRODIGY_native_train_seed0/checkpoint/state_dict_8000.ckpt \
  --output /dataMeR1/phil/gfm/prodigy-publickg-eval/log/publickg_native_eval_20260907 \
  --gpu 3

python -B scripts/experiments/setup/public_prodigy_kg/evaluate_cached.py \
  --phase quality --upstream /dataMeR1/phil/gfm/prodigy-public-upstream \
  --native-eval /dataMeR1/phil/gfm/prodigy-publickg-eval/log/publickg_native_eval_20260907 \
  --train-run /dataMeR1/phil/gfm/prodigy-publickg-native/log/publickg_train_20260907 \
  --output /dataMeR1/phil/gfm/prodigy-publickg-eval/log/publickg_quality_20260907 \
  --gpu 3

python -B scripts/experiments/setup/public_prodigy_kg/evaluate_cached.py \
  --phase roles --upstream /dataMeR1/phil/gfm/prodigy-public-upstream \
  --native-eval /dataMeR1/phil/gfm/prodigy-publickg-eval/log/publickg_native_eval_20260907 \
  --train-run /dataMeR1/phil/gfm/prodigy-publickg-native/log/publickg_train_20260907 \
  --quality-run /dataMeR1/phil/gfm/prodigy-publickg-eval/log/publickg_quality_20260907 \
  --output /dataMeR1/phil/gfm/prodigy-publickg-eval/log/publickg_roles_20260907 \
  --gpu 3
```

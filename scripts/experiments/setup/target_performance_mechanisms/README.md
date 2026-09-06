# Target-performance mechanism replay

Uses the existing `icl_arch_matrix` dataset construction, parameters, episode
fingerprint, and metric implementation. Does not change the production model.
Caches the actual collated test batches once per target; every model and input
intervention gets a fresh clone of the same batch. A second SHA-256 covers all
tensor values (including features). Stage observers must give bit-identical logits
to an unobserved forward. With 32 batches and seed offset 0, baseline full-model
metrics and episode hashes must match the reference lattice or the job fails.
The default metric tolerance is 1e-6. CPU/GPU float32 evaluation can reorder
near-tied scores: the first TwiBot cell differed by 1.70e-6 AUC with identical
accuracy/F1 and episode hash. `--auc-parity-atol 1e-5` explicitly permits this
small ranking-only portability error; accuracy/F1 stay at 1e-6 and per-cell
observed errors and the selected tolerance are saved. It is not exact-logit
parity with historical GPU outputs, which were not saved. Trace/no-trace logits
on the same device must still be bit-identical.

Run as a module from this worktree, in the prodigy environment:

```bash
python -m unittest scripts.experiments.setup.target_performance_mechanisms.test_replay
python -m scripts.experiments.setup.target_performance_mechanisms.replay \
  --model-list /dataMeR1/phil/gfm/prodigy-nm-pairs/log/nm_cls_lattice_20260905/model_list.tsv \
  --reference-tsv /dataMeR1/phil/gfm/prodigy-nm-pairs/log/nm_cls_lattice_20260905/classification_long.tsv \
  --output log/target_mechanisms/replay_unique_run_name \
  --specialists-only --include-random --save-embeddings --device 3 \
  --datasets covid_political,facebook_page_reference \
  --variants baseline,shuffle_context,center_features_only,no_background_edges,zero_support_relations,zero_label_text,bn_batch_encoder,bn_batch_meta,bn_batch_all \
  --dry-run
```

Remove `--dry-run` to execute. Use `--batch-count 1` for a smoke check (not a
result or aggregate parity check). Output directories must not already exist.
`DONE` is written only after all requested cells and parity checks complete.

Baseline probes apply cosine nearest-prototype and ridge (unit-normalized
embeddings, fixed ridge regularization 1) to raw center/context/joint features,
SAGE center outputs, S pooled features, U outputs, and post-metagraph outputs.
They fit only episode supports. Post-metagraph probes are support-conditioned,
not label-independent frozen representations. Probe scores use the same global
class mapping and query set as full PRODIGY; no hyperparameters are test-selected.

`S` performs both convolution and learned center-plus-mean readout. `U` copies
that center to the synthetic supernode in the current configuration. This is
why tracing S's internal readout inputs and U's actual output matters.

Intervention meanings:

- `shuffle_context`: exchange non-center real-node feature rows within an episode;
  preserves centers, topology, sampled sizes, and episode feature multiset.
- `center_features_only`: replace real-node context features with that subgraph's
  center features; preserve synthetic nodes and topology. Not a size-matched
  retraining experiment, and not equivalent to removing the encoder.
- `no_background_edges`: remove S background edges; keep pooling context and U/M
  edges. Separates message passing from direct context pooling.
- `zero_support_relations`: erase the +/- support-label edge channel.
- `zero_label_text`: zero the input label vectors; retains support relations and
  the initial projection bias. The legacy CLI name is unchanged, but these
  original-feature runs use class-keyed deterministic random vectors, not
  semantic text embeddings.
- `zero_support_and_label_text`: erase both prompt-label channels. Erasing only
  support relations does not guarantee chance when input label vectors remain.
- `bn_batch_*`: use current batch moments at the selected BN layers, without
  updating weights or running buffers. **Transductive diagnostic**: includes
  test-query covariates and neighboring episodes in the batch. This is not a
  leakage-free adaptation benchmark or evidence for a deployable improvement.
- `meta_bias_normalized`: apply the metagraph attention output-projection bias
  once per destination, retaining attention and weights. Production applies the
  affine output projection to each weighted message, then sums; its bias is
  therefore multiplied by destination indegree. The control removes only the
  excess bias. For default self-loops, an example node has degree 31 during
  30-way NM but degree 3 in binary CLS; a label node has degree 91 versus 21
  (30x3 versus 2x10 supports). This is a cardinality-sensitivity diagnostic,
  not a validated model fix: the weights were trained with the original term.

All are mechanistic sensitivity tests, not causal estimates of why a training
source won. Their main purpose is to choose a controlled follow-up. Runtime data,
cached batches, and large trace exports remain in Tucker's ignored log tree;
compact results/findings belong in the name-aligned analysis folder.

## Exact-source sampling audit

`audit_source_episodes` uses the production graph builder and member sampler on
the stored all-nine `static_train` graph. It simulates 256 new episodes per source
by default, recording eligible anchors, rejection frequency, duplicate members,
feature/role statistics, and raw-feature NM probes. It does **not** claim to
recover the historical multiworker training stream. The sorted member policy,
same retained members with shuffled support/query roles, and uniformly selected
unique walk endpoints are compared on the same successful anchors and walks.
Context summaries use a fresh sample of production-policy members, not a
counterfactual topology. No shared graph/checkpoint artifacts are modified.

```bash
python -m scripts.experiments.setup.target_performance_mechanisms.audit_source_episodes \
  --output log/target_mechanisms/source_audit_unique_name --dry-run
```

Run on Tucker CPU, with its own worktree and tmux session. Graph preprocessing is
memory-intensive; check host RAM and `/dev/shm` before execution. Newer PyTorch
versions memory-map the graph; Tucker's PyTorch 2.0 loads the full artifact.
Only the training adjacency is preprocessed. Runtime `.pt`
files contain exact simulated member IDs and corresponding feature rows; these
can support target-to-sampled-source coverage diagnostics later.

## Controlled member-selection training

`member_configs/` contains 24 prospective models: Ukraine/Hong Kong, three seeds,
and retained-endpoint policy (lowest IDs/uniform) crossed with role order
(ascending/shuffled). It is generated from the final-core recipe by:

```bash
python -m scripts.experiments.setup.target_performance_mechanisms.make_member_configs
```

The generator refuses an existing output directory. These configs use opt-in,
separate walk/retention/role random streams so changes in context sampling or
role randomization cannot shift subsequent positive walks. Legacy defaults
preserve the historical RNG and sorted selection. Validation/test always use
the legacy selector. Training state checkpoints include the private RNG states.
Consumed anchor/member IDs, roles, and context sizes are recorded in each run's
`data/consumed_episodes.jsonl.gz`, with hashes for matched-treatment verification.

Before substantive training, run the small CPU integration check **on Tucker**:

```bash
WANDB_MODE=offline python -m scripts.experiments.setup.target_performance_mechanisms.smoke_member_training \
  --output log/target_mechanisms/member_toy_unique_name
```

Then inspect the shared-graph plan (the example selects GPU 3; it must be free):

```bash
python -m scripts.experiments.setup.target_performance_mechanisms.run_member_training \
  --run-dir log/target_mechanisms/member_training_unique_name --gpus 3 --dry-run
```

The wrapper uses `experiments/run_shared_graph.py`, six models per GPU, four
loader workers per model, and a bounded total worker budget. CLI arguments can
change concurrency. It refuses active compute processes on selected GPUs,
insufficient host/shared RAM, or an existing run directory. Check tmux and pending
user jobs too: an apparently idle GPU may be reserved by a loading process.
Use a dedicated worktree and tmux. Do not start training until the selected
owned GPUs are available. `--smoke-steps 3` runs a separate GPU integration smoke;
its artifacts are not research results. The substantive budget remains 2500.

The prospective hypothesis, estimands, evaluation panel and validity gates are
recorded outside git at `../paper/planning/member_selection_intervention_2026-09-06.md`.

### Validated GPU-hidden CPU alternative

The 24-arm, 20-update concurrent smoke passed on Tucker at revision `75f0853f`.
Its automatic verifier confirmed common initial weights, consumed anchors,
retained sets within role pairs, and final walk RNG state. This path uses one
shared graph, six concurrent models, eight tensor threads and two loader workers
per model. It hides GPUs before importing PyTorch. It refuses insufficient RAM,
shared memory, an existing output directory, or more than a quarter of the host's
available logical CPU slots. Inspect current host load and other jobs as well.

```bash
python -m scripts.experiments.setup.target_performance_mechanisms.run_cpu_member_training \
  --run-dir log/target_mechanisms/member_cpu_unique_name \
  --models 6 --threads-per-model 8 --workers-per-model 2 --dry-run
```

For a separate smoke use `--smoke-steps 20` and remove `--dry-run`. For the full
2,500-update run remove both flags, from a dedicated frozen worktree in tmux.
All 24 arms must use the same training device; do not combine CPU and GPU arms
in the factorial contrast. CPU trajectories are not bitwise reruns of the old
GPU models. The prospective protocol records this execution amendment.

Both launchers must be followed by consumed-stream verification. The CPU launcher
does so automatically. The standalone command is:

```bash
python -m scripts.experiments.setup.target_performance_mechanisms.verify_member_training \
  --run-dir log/target_mechanisms/member_cpu_unique_name \
  --output log/target_mechanisms/member_cpu_unique_name/verified
```

Only a substantive run that passes all checks produces a validated
`verified/model_list.tsv` for the fixed-episode replay. Its primary checkpoint is
2,500 updates, regardless of intermediate target scores. Evaluate both episode
offsets 0 and 100003, with separate outputs, all five targets, and no historical
metric-parity requirement for these new weights. Check target fingerprints against
the established caches, and weight hashes across episode streams, in the analysis.

### Finite training-to-evaluation dependency

`finish_member_pipeline.py` can run in a separate frozen worktree/tmux session
while the existing CPU trainer completes. It does not launch or modify training.
It waits at most 12 hours for the substantive validity receipt, rejects failed or
smoke runs and revision mismatches, then replays all 24 models on both prescribed
episode streams using four CPU threads and hidden GPUs. It validates all 320
cached batch hashes against the established original/fresh caches and publishes
`input_validation.json` only after both complete. Example (use the actual training
revision, not the current evaluation worktree's HEAD):

```bash
python -m scripts.experiments.setup.target_performance_mechanisms.finish_member_pipeline \
  --training-run /dataMeR1/phil/gfm/prodigy-mechanisms-train/log/target_mechanisms/member_cpu_training_20260906 \
  --training-revision 75f0853f96272120a1e295dfbd38c04f2b71fe62 \
  --output log/target_mechanisms/member_evaluation_unique_name --dry-run
```

Remove `--dry-run` to start the finite continuation. Existing outputs are never
overwritten; a failed dependency stops with a preserved `pipeline.json` report.
The two replay logs and outputs remain underneath that unique output directory.

### Exact readout-weight intervention (exploratory, no training)

After the completed member-policy and exact-initialization analyses,
`build_readout_interventions.py` constructs two hybrids for every one of the
24 matched models. It swaps only `layer_list.0.reset_mlp_{c,m}.{weight,bias}`:
initial readout into terminal background, and terminal readout into initial
background. All other model tensors, including buffers and downstream weights,
must remain exactly equal to their declared background. This is not a claim
about the parameterless U operation or an additive decomposition of training.

```bash
python -m scripts.experiments.setup.target_performance_mechanisms.build_readout_interventions \
  --verified-training /dataMeR1/phil/gfm/prodigy-mechanisms-train/log/target_mechanisms/member_cpu_training_20260906/verified \
  --output log/target_mechanisms/readout_interventions_unique_name --threads 4 --dry-run
```

Remove `--dry-run` to construct the 48 frozen states, preserving all parents.
The output contains the exact tensor-provenance manifest and a `model_list.tsv`
accepted by the existing replay. Evaluate all five targets, `--variants baseline`,
`--device 123 --threads 4`, and offsets 0 and 100003 in distinct outputs. Do not
use `--include-random` or select a checkpoint. Require complete inputs/weights
and unchanged upstream probe outputs before interpreting readout changes.
The full exploratory protocol is outside git in
`../paper/planning/readout_weight_intervention_2026-09-06.md`.

After both streams complete, run the independent saved-artifact audit from a
separate frozen checkout; never update the active replay worktree:

```bash
python -m scripts.experiments.setup.target_performance_mechanisms.verify_readout_replay \
  --interventions /dataMeR1/phil/gfm/prodigy-mechanisms-readout/log/target_mechanisms/readout_interventions_20260906 \
  --original /dataMeR1/phil/gfm/prodigy-mechanisms-readout/log/target_mechanisms/readout_replay_original_20260906 \
  --fresh /dataMeR1/phil/gfm/prodigy-mechanisms-readout/log/target_mechanisms/readout_replay_fresh_20260906 \
  --output log/target_mechanisms/readout_verification_unique_name
```

All terminal/initial/cue reference roots are overrideable. The auditor rereads
every saved hybrid and both donors, verifies all tensor identities, and compares
153,600 upstream prediction tensors exactly (ten decoders, 32 batches, 480 cells).
It also exports the planned saved-logit cue comparisons for all hybrids and their
backgrounds. Its receipt is written only after every target and both streams
pass; no query labels are used for fitting or selecting a comparison.
# Readout training constraint (prospective follow-up, 6 September)

`run_readout_training.py` uses the existing shared-CPU-graph supervisor for 18
new runs: Ukraine/Hong Kong/COVID × seeds 0/1/2 × free/frozen initial readout.
Only the four `layer_list.0.reset_mlp_{c,m}.{weight,bias}` tensors are frozen;
the remainder of the network trains normally. No model algebra or sampling
implementation is changed. The original lowest-ID/sorted policy and private
sampler streams are retained. The primary predicted effect is **full-model
Facebook AUC improvement**, not merely a readout-probe gain. Report every seed
and source, all five targets, and both streams. This is discovery on previously
inspected targets, not untouched-domain confirmation.

The setup hook excludes the four frozen tensors from the unused optimizer and
checks them after every update. A forward pre-hook hashes every training-input
tensor before the model mutates it. The verifier requires matching complete
input streams and initializations within each pair, declared configurations,
source labels, final sampler states, maintained frozen weights at all saved
checkpoints, and updated control/readout and other network weights. The three
seeds are the replications; sources and episode streams are not extra seeds.

Run only in a new, frozen Tucker worktree, with the documented `prodigy` conda
environment and GPUs hidden. Unit checks are local; actual smoke training runs
on Tucker. An eight-update full-graph smoke validates the 18-arm launcher before
the substantive 2,500-update experiment. Use separate output paths; do not reuse
smoke outputs as research results.

For detached launches from an already-activated parent, invoke
`/home/mhchu/miniconda3/envs/prodigy/bin/python` explicitly. Prepending conda's base
bin followed by re-activating the already active environment can leave the base
Python first on PATH. This was reproduced during a failed import-only launch;
using the explicit environment interpreter avoids that ambiguity for this run
and for child verifiers/evaluators, which inherit `sys.executable`.

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline python -m \
  scripts.experiments.setup.target_performance_mechanisms.smoke_readout_training \
  --output log/target_mechanisms/readout_constraint_toy_20260906
CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline python -m \
  scripts.experiments.setup.target_performance_mechanisms.run_readout_training \
  --run-dir log/target_mechanisms/readout_constraint_smoke_20260906 --smoke-steps 8
CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline python -m \
  scripts.experiments.setup.target_performance_mechanisms.run_readout_training \
  --run-dir log/target_mechanisms/readout_constraint_training_20260906 --evaluate
```

`--dry-run` prints the plan without loading a graph. `--evaluate` is forbidden
for smoke runs; for substantive runs it executes only after all 18 validity
gates pass and verifies both target streams against established cached inputs.
`finish_readout_training.py` can perform that finite evaluation continuation
separately with explicit training/output/reference paths. All runtime files and
full training-input fingerprints stay private on Tucker until compact evidence
is collected. No public push is required.

## Complete mixture-complementarity diagnostic

`prepare_mixture_complementarity.py` snapshots and validates the historical
54-model, five-target, seed-zero CLS lattice. It requires all singletons, pairs,
and leave-one-out compositions at update 2,500, with matching checkpoint/source
inventories. Snapshot data and file digests live in the analysis folder's
`data/mixture_complementarity_inputs/`; newer role-corrected runs are not included.

`run_mixture_complementarity.py` verifies all 54 finite, architecture-compatible
checkpoint states, then replays all 45 mixtures on both complete fixed test
streams using the existing stage replay. It requires all 7,650 rows, 450 full-
model cells, exact cached-input checks, and the 225 original-stream reference
parity checks before publishing a completion receipt. No new model training or
GPU is needed. Run it in a separate frozen Tucker worktree, never the worktree
holding the active readout-training job:

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline \
  /home/mhchu/miniconda3/envs/prodigy/bin/python -m \
  scripts.experiments.setup.target_performance_mechanisms.run_mixture_complementarity \
  --inputs scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data/mixture_complementarity_inputs \
  --numerical-audit scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data/mixture_numerical_audit/audit.json \
  --output log/target_mechanisms/mixture_complementarity_20260906_v2 --threads 4
```

Use `--dry-run` before loading weights or data. Equal-probability constituent
averaging is the declared primary comparison; equal-logit averaging is secondary.
No target-selected weights or checkpoints. Foreign pair and foreign LOO results
stay separate. Extra ensemble compute, one training seed, and prior target
inspection limit the interpretation: this is not a causal interference test or
an untouched-domain prediction study.

After the complete replay, `analyze_mixture_predictions.py` reads saved logits
and the cached input tensors, recomputes all 540 constituent/mixture metrics,
and creates the fixed ensembles and unanimous/mixed error strata. It checks every
batch hash and the production global/episode-local label mapping. An incomplete
replay is rejected. Run this from an idle, appropriately revised worktree, not by
updating a worktree whose training or replay is still active:

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline \
  /home/mhchu/miniconda3/envs/prodigy/bin/python -m \
  scripts.experiments.setup.target_performance_mechanisms.analyze_mixture_predictions \
  --replay /dataMeR1/phil/gfm/prodigy-mechanisms-complement/log/target_mechanisms/mixture_complementarity_20260906_v2 \
  --output log/target_mechanisms/mixture_complementarity_predictions_20260906 --threads 4
```

The output consists of compact JSON tables and provenance; large prediction and
input tensors stay on Tucker. Copy the compact tables into the analysis folder's
`data/mixture_complementarity_predictions/` and run its
`analyze_mixture_complementarity.py` to reproduce the summaries. Original scalar
metric reproduction is an integrity gate, not a new independent evaluation.

The initial mixture replay was preserved after a suspension-target AUC guard
failed. The direct audit in `data/mixture_numerical_audit/` identifies exactly
one CPU/GPU score tie, with all decisions unchanged and three GPU repeats
matching the original metrics. Pass its `audit.json` via `--numerical-audit` to
the mixture runner, using a new output directory (the continuation is
`mixture_complementarity_20260906_v2`). This creates an explicitly aligned
reference **copy**, changes only the audited AUC cell, and retains both original
and CPU values. General replay tolerances remain unchanged. Completion records
separate 224 strict original mixture cells from one individually audited numerical
cell. The prediction analysis propagates this evidence to its compact export;
the local aggregate analysis checks it against the untouched historical table.

## Corrected-sampler checkpoint reuse

`run_corrected_sampler_replay.py` reads the nine completed seed-zero singleton
runs in `/dataMeR1/phil/gfm/prodigy-roleexposure`. Their source pipeline stopped
after training because a separate NM evaluation fingerprint ledger was missing;
this helper neither repairs nor restarts that pipeline. It verifies all 36 saved
100/300/900/2500 states against their training sidecars, optimizer steps, effective
configs, source restriction and recorded training revision. It then replays all
five classification targets, 17 decoders and both established episode streams
on CPUs, checking exact input correspondence. Run in a separate frozen worktree:

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline \
  /home/mhchu/miniconda3/envs/prodigy/bin/python -m \
  scripts.experiments.setup.target_performance_mechanisms.run_corrected_sampler_replay \
  --output log/target_mechanisms/corrected_sampler_replay_20260906 --threads 4
```

Use `--dry-run` before launch. The corrected selector shuffles before truncation,
changing retention and role order jointly and consuming randomness differently.
This reuse is not the matched three-seed factorial, a role-only causal comparison,
or an untouched-target validation. Require all 6,120 rows before comparing with
the historical trajectories; do not choose checkpoints on target performance.

## Specialist ensemble training-update budgets

`analyze_mixture_budget_predictions.py` requires completed historical trajectory
and mixture exports, then reads saved predictions only. It verifies all 810
distinct model/step/target/stream outputs and constructs all four specialist-step
ensembles for every pair/LOO mixture (1800 comparisons, 5400 error strata).
Before reading predictions, it resolves all 54 actual saved training configs
against their completed run logs/results and exact checkpoint paths, checking
source restrictions, four-episode batches, the 2500-update budget and explicit
saved-step schedule. Compact config hashes and checked fields are retained.
Every terminal-step result must reproduce the completed complementarity analysis.
The same production probability/logit and error-accounting functions are reused.

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline \
  /home/mhchu/miniconda3/envs/prodigy/bin/python -m \
  scripts.experiments.setup.target_performance_mechanisms.analyze_mixture_budget_predictions \
  --output log/target_mechanisms/mixture_budget_predictions_20260906 --threads 4 --dry-run
```

After the dry run passes, remove `--dry-run` and run in its own frozen Tucker
worktree/tmux session. Output/data/trajectory/mixture paths are overrideable.
Copy completed compact JSON tables to the analysis folder's
`data/mixture_budget_predictions/` and run `analyze_mixture_budget.py` locally.
No new training or model forward is performed. This compares saved-update and
episode budgets, not measured FLOPs or wall time. Ensembles retain 2x/8x inference
models; results are not a causal interference test or untouched-target evidence.
# Episode-cardinality attention diagnostic (2026-09-06)

`run_episode_cardinality.py` reuses the nine historical specialists' saved
pre-metagraph embeddings. It applies the production metagraph and cosine decoder
under seven fixed attention/bias multiplicity conditions, without encoder passes
or new training. Every baseline suffix and restored suffix must reproduce saved
logits bit-exactly; cached post-metagraph embeddings must also match exactly.
This is a multiplicity diagnostic, not an actual 30-class evaluation.

The prespecified primary is joint attention-only restoration on foreign donors
for both Facebook and TwiBot, positive and better than its reciprocal-direction
control on both streams. The full 630-cell grid and all controls are retained.
The prospective design is in the sibling paper planning directory:
`planning/episode_cardinality_diagnostic_2026-09-06.md`.

Run in an isolated Tucker worktree with GPUs hidden and the explicit `prodigy`
interpreter. Add `--dry-run` first; output must not already exist.

```bash
CUDA_VISIBLE_DEVICES="" WANDB_MODE=offline OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
/home/mhchu/miniconda3/envs/prodigy/bin/python -m \
  scripts.experiments.setup.target_performance_mechanisms.run_episode_cardinality \
  --original-roots \
    /dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms/specialist_cpu_20260906 \
    /dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms/specialist_cpu_tail_20260906 \
  --fresh-roots \
    /dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms/fresh_stage_cpu_20260906 \
  --training-audit \
    /dataMeR1/phil/gfm/prodigy-mechanisms-budget/log/target_mechanisms/mixture_budget_predictions_20260906/training_budget_audit.json \
  --output log/target_mechanisms/episode_cardinality_20260906 --threads 8
```

Analyze a completed, verified grid with
`scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.analyze_episode_cardinality`
using `--results <run-directory> --output <analysis-data-directory>`.

## Bounded CPU numerical replay

`audit_update_numerics.py` diagnoses update reproducibility without changing
production defaults or completed models. In an isolated Tucker worktree, invoke
the module with `--output <new-directory> --steps 4 --threads 8 --feature-dim 768
--localize-decoder --initial-checkpoint <actual-step-zero-training-state>`.
It runs default/deterministic/decoder-only-index-select computation, plain and
with the free audit hook, twice each. It caches four synthetic NM batches and
compares complete logits, gradients and state after every update. Replacement
decoder forwards must match original forwards exactly.

Add `--real-training-input-audit <completed-Ukraine-seed0-constraint_inputs.jsonl.gz>`
to load the real merged source artifact with the original 2500-batch/two-worker
contract. This mode requires the actual initial checkpoint, original resume
parameter contract, and exact full prefix input hashes before replay. It keeps
the 512-GiB available-host / 200-GiB shared-memory guards and restricts every
sampled real node to Ukraine. No target evaluation is run. The completed compact
receipts are analyzed downstream; full batches stay on Tucker.

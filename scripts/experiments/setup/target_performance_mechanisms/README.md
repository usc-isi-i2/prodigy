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
- `zero_label_text`: zero raw label-text embeddings; retains support relations.
- `zero_support_and_label_text`: erase both prompt-label channels. Erasing only
  support relations does not guarantee chance when label text is still present.
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

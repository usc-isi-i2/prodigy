# Target-performance mechanism replay

Uses the existing `icl_arch_matrix` dataset construction, parameters, episode
fingerprint, and metric implementation. Does not change the production model.
Caches the actual collated test batches once per target; every model and input
intervention gets a fresh clone of the same batch. A second SHA-256 covers all
tensor values (including features). Stage observers must give bit-identical logits
to an unobserved forward. With 32 batches and seed offset 0, baseline full-model
metrics and episode hashes must match the reference lattice or the job fails.

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
- `bn_batch_*`: use current batch moments at the selected BN layers, without
  updating weights or running buffers. **Transductive diagnostic**: includes
  test-query covariates and neighboring episodes in the batch. This is not a
  leakage-free adaptation benchmark or evidence for a deployable improvement.

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
memory-intensive; check host RAM and `/dev/shm` before execution. The graph is
memory-mapped and only the training adjacency is preprocessed. Runtime `.pt`
files contain exact simulated member IDs and corresponding feature rows; these
can support target-to-sampled-source coverage diagnostics later.

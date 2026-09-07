# Native public KG reproduction and paired replay

`run_native.py` uses the pinned upstream recipe without modifying its source.
Run it with `--help` for the required upstream/assets/output locations. It is
dry-run by default; `--execute` performs the selected phase. The nominated
evaluation checkpoint is `state_dict_8000.ckpt`, not a target-selected checkpoint.
Keep evaluation in a separate worktree from live training.

Evaluation capture schema 2 saves each actual pre-forward input, native output,
and pre-forward buffers/module modes/RNG state, with file hashes. Captures are
trusted local PyTorch artifacts, not a format for loading untrusted downloads.

`paired_replay.py` supplies the parity gate used before an intervention:

1. Construct the native model and load the exact checkpoint identified by the
   evaluation protocol hash. Do not attach the capture observer to this model.
2. Load a schema-2 record with `load_record`.
3. Call `replay_native` on the same device configuration. Exact logits and labels
   are required by default. Any justified nonzero GPU tolerance must be recorded.
4. Run each intervention under `isolated_forward_state(model, record_state)` on
   a fresh clone of the original inputs. Restore state separately for each arm.
   Never update parameters. The context restores caller buffers/modes/RNG even
   after errors; it does not undo parameter mutations or manage intervention hooks.

Native inference retains training-mode batch normalization. Do not insert an
`eval()` call. The replay runs without gradients, matching native evaluation.
CPU fixture parity is not evidence of native GPU parity.

## Public role intervention

`role_interventions.py` implements the support-background-edge-removal donor and
single-layer support K/V projection transplants. It preserves sampled nodes,
relation endpoints, head/tail features, pooling edges and episode labels. Query
background-edge removal is available as a role comparison. The modified Batch
is for direct forward use, not `to_data_list()` (its original slice metadata is
not rebuilt). Unrecognized background-edge payloads fail closed.

Before observing public target outcomes, nominate the **first metagraph layer**
for the primary K-only, V-only and joint support transplants. This is the layer
whose support inputs directly receive the graph encoder's outputs, matching the
discovery intervention's location. Capture each donor from the same immutable
episode and pre-forward state. The support mask must include false entries for
every label row. Do not use the second layer as a replacement primary endpoint
if the first layer's result is unfavorable.

Unlike the discovery model, native train-mode BN and two metagraph layers can
change queries. A value-only transplant fixes its local projection's Q/K blocks,
not all downstream attention or query representations. Joint support K/V need
not reproduce the whole context-removal endpoint because the donor can also
change non-support rows and residual paths. Report these differences rather than
enforce an invalid equivalence. Final-class-reference substitution with native
final queries remains necessary to separate the classifier and query effects.
`episode_mechanism.py` runs all six arms and crosses final decoder inputs to
measure reference-only and query-only changes. `run_mechanism.py` attaches this
experiment to each native evaluation forward. Its recursive replays are excluded
from the native episode stream and restore the model's post-native state. Each
episode artifact saves exact native inputs/state/output and the paired results.
The parent `execution_status.json` is the completion authority; a partial index
does not prove all 500 episodes completed.

Invoke the runner as a module (dry-run unless `--execute` is supplied):

```bash
python -m scripts.experiments.setup.public_prodigy_kg.run_mechanism \
  --upstream /dataMeR1/phil/gfm/prodigy-public-upstream \
  --root /dataMeR1/phil/data/prodigy_public_original \
  --output /dataMeR1/phil/gfm/prodigy-publickg-paired-state/log/publickg_mechanism \
  --checkpoint /dataMeR1/phil/gfm/prodigy-publickg-native/log/publickg_train_20260907/state/Wiki_PT_PRODIGY_native_train_seed0/checkpoint/state_dict_8000.ckpt \
  --gpu 3
```

This implementation has passed a tiny fixture with the pinned upstream decoder
and two-layer metagraph, including two consecutive native episodes with versus
without online interventions. The background encoder/pooling in that fixture are
test components. Full native GPU/data validation remains required.

Light checks (no datasets or training):

```bash
python -m unittest scripts.experiments.setup.public_prodigy_kg.test_run_native -q
python -m scripts.experiments.setup.public_prodigy_kg.test_forward_state
python -m scripts.experiments.setup.public_prodigy_kg.test_role_interventions
python -m scripts.experiments.setup.public_prodigy_kg.test_episode_mechanism /path/to/pinned/upstream
```

The second command needs PyTorch and NumPy; use the `prodigy` environment.

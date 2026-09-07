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
These helpers do not yet implement the KG support-neighborhood donor or K/V
interventions, and CPU fixture parity is not evidence of native GPU parity.

Light checks (no datasets or training):

```bash
python -m unittest scripts.experiments.setup.public_prodigy_kg.test_run_native -q
python -m scripts.experiments.setup.public_prodigy_kg.test_forward_state
```

The second command needs PyTorch and NumPy; use the `prodigy` environment.

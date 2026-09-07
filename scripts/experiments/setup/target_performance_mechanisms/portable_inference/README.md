# Portable inference candidate build and real-data validation

Builds an explicit dependency slice of the current model and intervention code,
not an alternate reimplementation. The `template/` supplies a direct model
constructor, tensor-only schema, complete inference runner and synthetic tests.
`build.py` adds production model files and verbatim diagnostic functions with
source hashes. It does not distribute datasets, trained weights or dependencies.

Local build/test (lightweight model checks only):

```bash
/opt/homebrew/bin/python3.11 scripts/experiments/setup/target_performance_mechanisms/portable_inference/build.py --output /private/tmp/new-inference-candidate
cd /private/tmp/new-inference-candidate
/Users/philipp/miniconda3/envs/prodigy/bin/python -m unittest test_contracts
```

On the isolated Tucker worktree, the launcher builds the archive, extracts it
outside the checkout, runs synthetic tests, exports a private tensor input
bundle, smoke-tests three targets, then evaluates the complete final-checkpoint
panel from the extracted package. Each stage fails before the next if invalid.
No source training, graph construction, GPU use or public upload occurs.

```bash
tmux new-session -d -s mechanism-portable-inference 'cd /dataMeR1/phil/gfm/prodigy-role-topology && bash scripts/experiments/setup/target_performance_mechanisms/portable_inference/run_tucker.sh /dataMeR1/phil/gfm/inference-reproduction-20260907'
```

The export adapter is the only component that reads the trusted research
checkout and old PyG object caches. It checks the original batch/weight hashes,
removes account identifiers and confirms constructor/input-adapter parity for
all 60 model/target/stream first batches. Features remain private. The extracted
runtime subsequently loads only tensors/primitives and has no research imports.

The final run covers six actual update-2500 checkpoints, five targets, two
streams and 45 conditions: 2,700 reported metric cells. Historical reference
logits/metrics exist for 43 conditions per input/model (2,580 cells); raw and
encoder prototype readouts are additional diagnostics. This package supports
compatible checkpoints at other steps, but this declared real-data validation
does not claim to replay all five saved steps or the historical nine-source map.

Use `full/DONE.json`, not elapsed time or a created directory, as completion
evidence. `smoke/DONE.json` is explicitly partial. Failed observation is not
permission to restart; inspect the existing tmux session and logs. No automatic
overwrite or resume occurs. Licensing, access and public release decisions are
separate from numerical portability; see the package notice.

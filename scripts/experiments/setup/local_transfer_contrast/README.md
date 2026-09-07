# Local transfer contrast

This experiment compares two fixed single-source PRODIGY checkpoints on exactly
the same target query occurrences. The discovery triple was selected from the
three-seed final-core matrix before inspecting example-level outcomes:

- target A: `facebook_page_reference` (150,000 nodes);
- strong source B: `twibot20` (162,990 nodes);
- weak source C: `election2020` (78,932 nodes).

The exporter reuses the completed role-context replay. It verifies cached batch,
label, query-order, raw-input, baseline-logit, and model-weight identities before
writing query-level inputs. It runs no training or model forward passes.

On Tucker, from an isolated checkout of this branch:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
python -m scripts.experiments.setup.local_transfer_contrast.export_contrast \
  --role-root /dataMeR1/phil/gfm/prodigy-mechanisms-role/log/role_context_20260906 \
  --output log/local_transfer_contrast/facebook_twibot_vs_election
```

The original episode stream is for exploratory discovery. The fresh stream is
reserved for validation of frozen clusters and routers. Both streams reuse the
same target domain and checkpoints; they are not independent training seeds.

Selected loss extremes can be reconstructed from the exact cached episodes and
then joined to the exact description text that produced each stored embedding.
Run `inspect_examples.py extract` in `prodigy`, then `inspect_examples.py
hydrate` in `bio-embeddings-v001`. The selection is saved before any text is
read; hydrated text stays in ignored experiment output rather than git.

Analysis lives under
`scripts/experiments/analysis/graphs/transfer_prediction/local_transfer_contrast/`.

## Independent-checkpoint replication

After constructing a 9-row singleton model list for each archived final-core
checkpoint seed, launch one fixed-episode replay per seed and stream from separate
tmux sessions:

```bash
bash scripts/experiments/setup/local_transfer_contrast/run_seed_replay_tucker.sh 1 original 0
bash scripts/experiments/setup/local_transfer_contrast/run_seed_replay_tucker.sh 1 fresh 1
bash scripts/experiments/setup/local_transfer_contrast/run_seed_replay_tucker.sh 2 original 2
bash scripts/experiments/setup/local_transfer_contrast/run_seed_replay_tucker.sh 2 fresh 3
```

The same launcher accepts seed 0. We rerun it under the identical current
code/hardware path when an older cache is not tensor-identical; this avoids mixing
different sampled neighborhoods into a checkpoint-seed comparison.

The target sampling seed remains zero for every checkpoint seed so the replication
isolates training stochasticity. The fresh stream uses the same frozen 100003 offset
as the seed-0 analysis. CUDA trace parity is explicitly guarded at `1e-5` because
scatter reduction order can change logits at the low `1e-6` scale.

Once both streams finish, `export_seed_replays.py` verifies their input tensors
against the seed-0 caches and emits the same `original.pt`/`fresh.pt` contract used
by the health and routing analyses. Query-level exports remain ignored runtime
artifacts; only aggregate statistics belong in git.

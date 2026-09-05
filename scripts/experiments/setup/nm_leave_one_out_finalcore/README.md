# NM leave-one-source-out final-core sweep

This setup appends the true leave-one-source-out complement to the seed-zero pair
sweep. It trains nine models: each model receives eight of the nine final-core
sources and excludes exactly one. Its sole production evaluation target is that
excluded source.

## Frozen contract

- one training seed (`0`)
- nine training mixtures (all sources minus one)
- 2,500 optimizer steps / 10,000 episodes per model
- final checkpoint `state_dict_2500.ckpt`
- matching `training_state_2500.ckpt` with optimizer, RNG, and sampler state
- shared CPU graph fast trainer, GPUs 0--3, two active trainers per GPU
- W&B offline
- nine held-out cells, each with the frozen 512-episode NM test stream
- `static_train` message-passing edges and `static_test` held-out positives
- accuracy, macro-F1, and macro one-vs-rest ROC-AUC
- strict comparison to the archived final-core episode fingerprint ledger

The first eight evaluation workers run concurrently, two per GPU. The ninth runs
afterward on GPU 0. This limits peak host memory while keeping all four owned GPUs
busy during the main wave.

## Tucker commands

Use the dedicated worktree `/dataMeR1/phil/gfm/prodigy-nm-loo` on branch
`codex/nm-loo-fast`.

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export WANDB_MODE=offline

python scripts/experiments/setup/nm_leave_one_out_finalcore/make_configs.py --replace
bash scripts/experiments/setup/nm_leave_one_out_finalcore/run_training_tucker.sh
python scripts/experiments/setup/nm_leave_one_out_finalcore/verify_training.py \
  --run-dir log/nm_leave_one_out_finalcore/shared_seed0_20260904
bash scripts/experiments/setup/nm_leave_one_out_finalcore/run_evaluation_tucker.sh
```

To append the entire pipeline after the pair sweep's strict evaluation receipt:

```bash
tmux new-session -d -s nmloo \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; cd /dataMeR1/phil/gfm/prodigy-nm-loo; bash scripts/experiments/setup/nm_leave_one_out_finalcore/run_after_pairs_tucker.sh > log/nm_leave_one_out_finalcore_pipeline_orchestrator.log 2>&1'
```

The chained launcher refuses to start if pair training fails or the pair evaluation
session exits without its strict completion receipt. It also waits for GPUs 0--3 to
be released before starting the LOO trainer.

For a non-mutating launch preview:

```bash
DRY_RUN=1 bash scripts/experiments/setup/nm_leave_one_out_finalcore/run_training_tucker.sh
DRY_RUN=1 bash scripts/experiments/setup/nm_leave_one_out_finalcore/run_evaluation_tucker.sh
```

## Outputs

Training receipts, effective configs, checkpoints, and offline W&B runs are under
`log/nm_leave_one_out_finalcore/shared_seed0_20260904/`. Strict evaluation outputs
are under `log/nm_leave_one_out_finalcore_eval/production/bs32/`, including:

- `results/seed_0/<model>/<heldout>.json`
- `summary/loo_heldout_metrics.tsv`
- `summary/loo_heldout_metrics.csv`
- `summary/completeness.json`
- `complete_utc.txt`, written only after all nine cells validate

The shared-run manifest records the exact Git revision, effective parameters,
physical GPU mapping, and timing. `verify_training.py` requires both the weight-only
and full training-state checkpoint for every model. With multiple data-loader
workers, the saved optimizer/RNG/sampler state is retained but exact mid-epoch replay
is not claimed.

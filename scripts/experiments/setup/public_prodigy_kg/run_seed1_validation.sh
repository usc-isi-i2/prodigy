#!/usr/bin/env bash
# Run once after the nominated seed-one checkpoint exists. Never waits or trains.
set -euo pipefail
if [[ "${1:-}" != "--execute" && $# -ne 0 ]]; then
  echo 'Usage: bash run_seed1_validation.sh [--execute]' >&2
  exit 2
fi
if [[ $# -gt 1 ]]; then exit 2; fi
validation_execute="${1:-}"
validation_repo="${VALIDATION_REPO:-/dataMeR1/phil/gfm/prodigy-publickg-paired-state}"
validation_training="${VALIDATION_TRAINING:-/dataMeR1/phil/gfm/prodigy-publickg-seed1/log/publickg_train_seed1_20260907}"
validation_upstream="${VALIDATION_UPSTREAM:-/dataMeR1/phil/gfm/prodigy-public-upstream}"
validation_data="${VALIDATION_DATA:-/dataMeR1/phil/data/prodigy_public_original}"
validation_gpu="${VALIDATION_GPU:-3}"
validation_tag="${VALIDATION_TAG:-20260907}"
validation_checkpoints="$validation_training/state/Wiki_PT_PRODIGY_native_train_seed1/checkpoint"
validation_source="$validation_repo/log/publickg_readout_seed1_$validation_tag"
validation_controls="$validation_repo/log/publickg_controls_seed1_$validation_tag"
validation_trajectory="$validation_repo/log/publickg_trajectory_seed1_$validation_tag"
validation_crossover="$validation_repo/log/publickg_crossover_seed1_$validation_tag"
validation_normalization="$validation_repo/log/publickg_normalization_seed1_$validation_tag"

validation_run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "$validation_execute" == '--execute' ]]; then "$@" --execute; fi
}

if [[ "$validation_execute" == '--execute' ]]; then
  # Fail before making any output if a nominated prerequisite is missing.
  for validation_step in 2000 4000 8000; do
    test -f "$validation_checkpoints/state_dict_$validation_step.ckpt"
  done
  test -f "$validation_training/initial_model.ckpt"
  for validation_output in "$validation_source" "$validation_controls" "$validation_trajectory" "$validation_crossover" "$validation_normalization"; do
    if [[ -e "$validation_output" ]]; then
      echo "Refusing to overwrite: $validation_output" >&2
      exit 1
    fi
  done
  export PATH="/home/mhchu/miniconda3/bin:$PATH"
  source /home/mhchu/miniconda3/etc/profile.d/conda.sh
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export WANDB_MODE=offline
  cd "$validation_repo"
fi

# Evaluation RNG remains zero and sampler offset100003, matching seed zero.
# Only the training checkpoint/initialization differ; no target-best selection.
validation_run python -m scripts.experiments.setup.public_prodigy_kg.run_readout_replication \
  --upstream "$validation_upstream" --root "$validation_data" \
  --output "$validation_source" --checkpoint "$validation_checkpoints/state_dict_8000.ckpt" --gpu "$validation_gpu"
validation_run python -m scripts.experiments.setup.public_prodigy_kg.pretraining_control \
  --source "$validation_source" --initialization "$validation_training/initial_model.ckpt" \
  --upstream "$validation_upstream" --output "$validation_controls" --gpu "$validation_gpu"
validation_run python -m scripts.experiments.setup.public_prodigy_kg.run_trajectory \
  --source "$validation_source" --checkpoint-dir "$validation_checkpoints" \
  --upstream "$validation_upstream" --output "$validation_trajectory" --gpu "$validation_gpu"
validation_run python -m scripts.experiments.setup.public_prodigy_kg.run_checkpoint_crossover \
  --source "$validation_source" --checkpoint-dir "$validation_checkpoints" \
  --trajectory "$validation_trajectory" --upstream "$validation_upstream" \
  --output "$validation_crossover" --gpu "$validation_gpu" --replication
validation_run python -m scripts.experiments.setup.public_prodigy_kg.run_normalization_sensitivity \
  --source "$validation_source" --checkpoint-dir "$validation_checkpoints" \
  --trajectory "$validation_trajectory" --upstream "$validation_upstream" \
  --output "$validation_normalization" --gpu "$validation_gpu"

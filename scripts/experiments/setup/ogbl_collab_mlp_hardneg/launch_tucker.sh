#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: launch_tucker.sh GPU SEED}"
SEED="${2:?usage: launch_tucker.sh GPU SEED}"
ROOT="${ROOT:-/dataMeR1/phil/gfm/ogbl_collab_mlp_hardneg/hardneg_v1}"
for SCORER in cosine interaction; do
  for POLICY in uniform hard8; do
    if [[ "${SCORER}_${POLICY}" == "cosine_uniform" ]]; then continue; fi
    python -u -m scripts.experiments.setup.ogbl_collab_mlp_hardneg.run \
      --scorer "${SCORER}" --negative-policy "${POLICY}" --seed "${SEED}" \
      --device "cuda:${GPU}" --out "${ROOT}/${SCORER}_${POLICY}/seed${SEED}"
  done
done

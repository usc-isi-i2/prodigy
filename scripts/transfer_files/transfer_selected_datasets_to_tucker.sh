#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="eibl@hpc-transfer1.usc.edu"
SRC_BASE="/scratch1/eibl/data"
DEST_BASE="/dataMeR2/phil/data"

DATASETS=(
  "covid19_twitter"
  "covid_masking"
  "covid_political"
  "ed"
  "election2020"
  "immigration_julia"
  "social_llm_covid"
  "ukr_rus_suspended"
)

for dataset in "${DATASETS[@]}"; do
  src="${REMOTE_HOST}:${SRC_BASE}/${dataset}/"
  dest="${DEST_BASE}/${dataset}"

  mkdir -p "${dest}"

  echo "Transferring ${dataset}..."
  rsync -avh --partial --info=progress2 \
    "${src}" \
    "${dest}"
done

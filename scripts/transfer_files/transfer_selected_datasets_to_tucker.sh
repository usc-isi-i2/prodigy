#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="eibl@hpc-transfer1.usc.edu"
SRC_BASE="/scratch1/eibl/data"
DEST_BASE="/dataMeR2/eibl/data"

DATASETS=(
  "covid19_twitter"
  "covid_political"
  "election2020"
  "midterm"
  "ukr_rus_suspended"
  "ukr_rus_twitter"
)

for dataset in "${DATASETS[@]}"; do
  src="${REMOTE_HOST}:${SRC_BASE}/${dataset}/graphs/"
  dest="${DEST_BASE}/${dataset}/graphs"

  mkdir -p "${dest}"

  echo "Transferring ${dataset}/graphs..."
  rsync -avh --partial --info=progress2 \
    "${src}" \
    "${dest}"
done

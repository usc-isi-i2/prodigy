#!/usr/bin/env bash

set -euo pipefail

DEST_ROOT="/dataMeR2/phil/data/ukr_rus_twitter/parquet"
SRC_ROOT="eibl@hpc-transfer1.usc.edu:/scratch1/eibl/data/ukr_rus_twitter/parquet/"

mkdir -p "${DEST_ROOT}"

rsync -avh --partial --info=progress2 \
  "${SRC_ROOT}" \
  "${DEST_ROOT}"

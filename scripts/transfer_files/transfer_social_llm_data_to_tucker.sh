#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="eibl@hpc-transfer1.usc.edu"
SRC_PATH="/project2/emiliofe_74/julie/social_llm_data"
DEST_ROOT="/dataMeR2/phil/data"

mkdir -p "${DEST_ROOT}"

rsync -avh --partial --info=progress2 \
  "${REMOTE_HOST}:${SRC_PATH}" \
  "${DEST_ROOT}"

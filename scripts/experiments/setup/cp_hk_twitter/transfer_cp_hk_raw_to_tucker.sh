#!/usr/bin/env bash
# Two-hop CP-HK raw-data transfer helper.
#
# Tucker cannot SSH back to CARC in the observed environment, so this runs from
# the local machine and streams each file CARC -> local SSH client -> Tucker.
set -euo pipefail

CARC_HOST="${CARC_HOST:-carc}"
TUCKER_HOST="${TUCKER_HOST:-tucker}"
CARC_DIR="${CARC_DIR:-/project2/emiliofe_74/data_backup/COSINE/2022CP-HK}"
TUCKER_RAW_DIR="${TUCKER_RAW_DIR:-/dataMeR1/phil/data/cp_hk_twitter/raw}"

files=(
  "an_cp-hk.twitter.v7-ground-truth.2020-04-07_2020-08-23.json.gz"
  "an_cp-hk.twitter.v7-ground-truth.2020-08-24_2020-09-13.json.gz"
)

ssh -o BatchMode=yes -o ConnectTimeout=8 "${TUCKER_HOST}" "mkdir -p '${TUCKER_RAW_DIR}'"

for file in "${files[@]}"; do
  echo "Transferring ${file}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${CARC_HOST}" "cat '${CARC_DIR}/${file}'" \
    | ssh -o BatchMode=yes -o ConnectTimeout=8 "${TUCKER_HOST}" \
        "cat > '${TUCKER_RAW_DIR}/${file}.tmp' && mv '${TUCKER_RAW_DIR}/${file}.tmp' '${TUCKER_RAW_DIR}/${file}'"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${TUCKER_HOST}" "ls -lh '${TUCKER_RAW_DIR}/${file}'"
done

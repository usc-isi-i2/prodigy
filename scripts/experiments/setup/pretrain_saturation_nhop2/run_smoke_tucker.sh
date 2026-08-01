#!/usr/bin/env bash
# Resource/wiring gate on the densest source graph. Its checkpoints are not evidence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
"${SCRIPT_DIR}/train_nm_tucker.sh" smoke_dense_election.yaml --device "${GPU}" "$@"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/submit_train3_from_train2_all.sh" midterm "${SCRIPT_DIR}/model_lists/train2_all_models.txt"

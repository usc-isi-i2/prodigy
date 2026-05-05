#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/submit_train3_same_dataset_all.sh" midterm "${SCRIPT_DIR}/train1_midterm_1p5m_model_list.txt"

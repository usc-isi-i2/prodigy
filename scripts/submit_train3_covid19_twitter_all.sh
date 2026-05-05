#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/submit_train3_same_dataset_all.sh" covid19_twitter "${SCRIPT_DIR}/train1_covid19_twitter_1p5m_model_list.txt"

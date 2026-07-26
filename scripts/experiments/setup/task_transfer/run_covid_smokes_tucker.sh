#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for task in nm cl fp; do
  bash "${SCRIPT_DIR}/train_covid_task_tucker.sh" "${task}" "$@"
done


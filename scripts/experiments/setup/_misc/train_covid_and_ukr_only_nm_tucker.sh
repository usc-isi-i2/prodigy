#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/covid_only/train_covid_only_nm_tucker.sh" "$@"
"${SCRIPT_DIR}/ukr_only/train_ukr_only_nm_tucker.sh" "$@"

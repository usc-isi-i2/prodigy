#!/usr/bin/env bash
# Run the pre-train graph sanity check in the prodigy conda env.
#   ./inspect_graphs_tucker.sh            # uses default Tucker paths
#   ./inspect_graphs_tucker.sh --ukr ... --covid ... --merged ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/inspect_graphs.py" "$@"

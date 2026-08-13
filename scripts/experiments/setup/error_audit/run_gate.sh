#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

"$PYTHON_BIN" scripts/eval/regression_probe.py --self-test
"$PYTHON_BIN" scripts/eval/pair_link_eval.py --self-test
"$PYTHON_BIN" scripts/experiments/analysis/evaluation/error_audit/tests/test_error_audit.py
"$PYTHON_BIN" scripts/experiments/analysis/evaluation/error_audit/tests/test_episode_export.py

bash -n \
  scripts/experiments/setup/error_audit/run_episode_audit_tucker.sh \
  scripts/experiments/setup/error_audit/run_regression_audit_tucker.sh \
  scripts/experiments/setup/error_audit/run_static_lp_audit_tucker.sh

"""Shared metric contract for final-core ROC-AUC evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


METRIC_CONTRACT = "accuracy_f1_macro_roc_auc_ovr_macro_v1"


def metric_sidecar_path(logging_dir: str | Path, target: str, step: int) -> Path:
    return Path(logging_dir) / f"metrics_test_{target}_step{step}.json"


def load_metric_sidecar(logging_dir: str | Path, target: str, step: int) -> dict[str, float]:
    path = metric_sidecar_path(logging_dir, target, step)
    if not path.is_file():
        raise FileNotFoundError(f"evaluation metric sidecar was not written: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "accuracy": f"test_{target}_accuracy",
        "f1_macro": f"test_{target}_f1",
        "roc_auc_ovr_macro": f"test_{target}_roc_auc",
    }
    metrics = {}
    for output_name, sidecar_name in expected.items():
        value = float(payload.get(sidecar_name, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"{path}: missing or non-finite {sidecar_name}")
        metrics[output_name] = value
    return metrics


def load_reference_fingerprints(
    path: Path | None, expected_targets: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_target = {row["target"]: row for row in rows}
    if len(rows) != len(expected_targets) or set(by_target) != set(expected_targets):
        raise ValueError(f"reference fingerprint ledger has the wrong targets: {path}")
    for target, row in by_target.items():
        for key in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            if len(row.get(key, "")) != 64:
                raise ValueError(f"{path}: target {target} has invalid {key}")
    return by_target

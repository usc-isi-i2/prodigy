from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


RUN_RE = re.compile(
    r"mscale_(?P<target>.+)_k(?P<k>[136])_subset(?P<subset>\d+)_w(?P<width>64|256|512)_existing_s0\.json"
)


def auc(row: dict) -> float:
    return float(row.get("auc", row.get("roc_auc_ovr_macro")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.ladder, newline="") as handle:
        ladder = [row for row in csv.DictReader(handle) if int(row["k"]) in (1, 3, 6)]
    expected = {
        (row["target"], int(row["k"]), int(row["subset"]), width): row
        for row in ladder for width in (64, 256, 512)
    }
    rows: dict[tuple[str, int, int, int], dict] = {}
    for row in ladder:
        if row["mode"] == "existing":
            key = (row["target"], int(row["k"]), int(row["subset"]), 256)
            rows[key] = {**row, "width": 256, "reused_width256": True}
    for path in Path(args.raw_root).glob("*.json"):
        match = RUN_RE.fullmatch(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        key = (match["target"], int(match["k"]), int(match["subset"]), int(match["width"]))
        rows[key] = {
            "target": key[0], "k": key[1], "subset": key[2], "width": key[3],
            "mode": "existing", "auc": auc(data),
            "checkpoint_step": int(data["checkpoint_step"]),
            "sources": "|".join(data["sources"]), "path": str(path),
            "reused_width256": False,
        }
    missing = sorted(set(expected) - set(rows))
    extra = sorted(set(rows) - set(expected))
    if missing or extra:
        raise ValueError(f"model-scale audit failed: missing={len(missing)} extra={len(extra)}")

    ordered = [rows[key] for key in sorted(rows)]
    fields = ["target", "k", "subset", "width", "mode", "auc", "checkpoint_step", "sources", "path", "reused_width256"]
    with open(output_root / "model_scale_auc.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(ordered)

    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in ordered:
        grouped[(row["target"], int(row["width"]), int(row["k"]))].append(auc(row))
    means = {(t, w, k): float(np.mean(v)) for (t, w, k), v in grouped.items()}
    slopes = []
    for target in sorted({row["target"] for row in ordered}):
        for width in (64, 256, 512):
            x = np.log([1, 3, 6])
            y = np.array([means[target, width, k] for k in (1, 3, 6)])
            slope = float(np.polyfit(x, y, 1)[0])
            slopes.append({"target": target, "width": width, "log_k_slope": slope,
                           "k1_auc": y[0], "k3_auc": y[1], "k6_auc": y[2]})
    with open(output_root / "model_scale_slopes.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slopes[0]))
        writer.writeheader(); writer.writerows(slopes)

    # Target fixed effects; log(width) and log(k) are centered before interaction.
    targets = sorted({row["target"] for row in ordered})
    target_index = {target: i for i, target in enumerate(targets)}
    log_k = np.log([int(row["k"]) for row in ordered]); log_k -= log_k.mean()
    log_w = np.log([int(row["width"]) for row in ordered]); log_w -= log_w.mean()
    fixed = np.zeros((len(ordered), len(targets) - 1))
    for i, row in enumerate(ordered):
        index = target_index[row["target"]]
        if index:
            fixed[i, index - 1] = 1
    design = np.column_stack((np.ones(len(ordered)), fixed, log_k, log_w, log_k * log_w))
    y = np.array([auc(row) for row in ordered])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    dof = len(y) - design.shape[1]
    covariance = (residual @ residual / dof) * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(covariance))
    interaction_index = design.shape[1] - 1
    interaction_t = beta[interaction_index] / se[interaction_index]
    audit = {
        "status": "complete", "cells": len(ordered), "expected_cells": 147,
        "targets": len(targets), "widths": [64, 256, 512], "mixture_sizes": [1, 3, 6],
        "interaction_log_k_log_width": float(beta[interaction_index]),
        "interaction_standard_error": float(se[interaction_index]),
        "interaction_p_value": float(2 * stats.t.sf(abs(interaction_t), dof)),
    }
    (output_root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

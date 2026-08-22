from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


GRAPHS = (
    "covid_political",
    "ukr_rus_suspended",
    "election2020",
    "twibot20",
    "facebook_page_reference",
    "cora",
    "pubmed",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for source in GRAPHS:
        for target in GRAPHS:
            path = Path(args.raw_root) / f"{source}__to__{target}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text())
            if data.get("status") != "complete":
                raise ValueError(f"incomplete result: {path}")
            if data.get("target") != target or data.get("sources") != [source]:
                raise ValueError(f"identity audit failed: {path}")
            rows.append({
                "source": source,
                "target": target,
                "seed": int(data["pretrain_seed"]),
                "checkpoint_step": int(data["checkpoint_step"]),
                "auc": float(data["roc_auc_ovr_macro"]),
                "f1_macro": float(data["f1_macro"]),
                "accuracy": float(data["accuracy"]),
                "target_split_hash": data["target_split_hash"],
                "result_path": str(path),
            })
    if len(rows) != 49 or len({(r["source"], r["target"]) for r in rows}) != 49:
        raise ValueError("corrected matrix must contain exactly 49 unique cells")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "complete",
        "cells": 49,
        "expected_cells": 49,
        "sources": list(GRAPHS),
        "targets": list(GRAPHS),
        "protocol": "heldout_training_edges_width256_existing_features_seed0",
    }
    (output.parent / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

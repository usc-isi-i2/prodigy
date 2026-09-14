from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--raw-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); files = sorted(Path(args.raw_root).glob("*.json"))
    if len(files) != 72: raise ValueError(f"expected 72 cells, found {len(files)}")
    rows, identities, hashes = [], set(), {}
    for path in files:
        data = json.loads(path.read_text())
        if data.get("status") != "complete" or data.get("budgets") != [1, 5, 10, 20, 50] or data.get("support_seeds") != list(range(10)): raise ValueError(path)
        identity = (data["target"], data["condition"], data["objective"], data["feature_mode"], data["pretrain_seed"])
        if identity in identities: raise ValueError(f"duplicate {identity}")
        identities.add(identity); hashes.setdefault(data["target"], set()).add(data["target_split_hash"])
        for result in data["results"]:
            rows.append({"target": data["target"], "condition": data["condition"], "objective": data["objective"],
                         "feature_mode": data["feature_mode"], "seed": data["pretrain_seed"],
                         "support_seed": result["support_seed"], "labels_per_class": result["labels_per_class"], "auc": result["roc_auc_ovr_macro"],
                         "f1_macro": result["f1_macro"], "accuracy": result["accuracy"],
                         "checkpoint_step": data["checkpoint_step"], "sources": "|".join(data["sources"])})
    if any(len(value) != 1 for value in hashes.values()): raise ValueError("split hashes differ")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if len(rows) != 3600: raise ValueError(f"expected 3600 metric rows, found {len(rows)}")
    audit = {"status": "complete", "evaluation_cells": 72, "metric_rows": len(rows), "budgets": [1, 5, 10, 20, 50], "support_seeds": list(range(10))}
    (output.parent / "audit.json").write_text(json.dumps(audit, indent=2) + "\n"); print(json.dumps(audit, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())

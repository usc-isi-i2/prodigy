from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ARMS = {
    "election": ["election2020"],
    "cora": ["cora"],
    "mixture_k3": ["covid_political", "ukr_rus_suspended", "election2020"],
    "mixture_k6": [
        "covid_political", "ukr_rus_suspended", "election2020",
        "facebook_page_reference", "cora", "pubmed",
    ],
}
PATTERN = re.compile(r"^(election|cora|mixture_k3|mixture_k6)_s([0-2])_step_(\d+)\.json$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    covered = set()
    for path in sorted(Path(args.raw_root).glob("*.json")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        arm, seed_text, requested_step = match.groups()
        seed = int(seed_text)
        data = json.loads(path.read_text())
        if data.get("status") != "complete" or data.get("target") != "twibot20":
            raise ValueError(f"result audit failed: {path}")
        if data.get("sources") != ARMS[arm]:
            raise ValueError(f"source audit failed: {path}")
        if int(data.get("pretrain_seed", -1)) != seed or int(data.get("probe_seed", -1)) != seed:
            raise ValueError(f"seed audit failed: {path}")
        if int(data["checkpoint_step"]) != int(requested_step):
            raise ValueError(f"checkpoint-step audit failed: {path}")
        covered.add((arm, seed))
        rows.append({
            "arm": arm,
            "seed": seed,
            "checkpoint_step": int(data["checkpoint_step"]),
            "auc": float(data["roc_auc_ovr_macro"]),
            "f1_macro": float(data["f1_macro"]),
            "accuracy": float(data["accuracy"]),
            "sources": "|".join(data["sources"]),
            "target_split_hash": data["target_split_hash"],
            "result_path": str(path),
        })
    expected = {(arm, seed) for arm in ARMS for seed in range(3)}
    if covered != expected:
        raise ValueError(f"missing arm-seed trajectories: {sorted(expected - covered)}")
    for key in expected:
        points = [row for row in rows if (row["arm"], row["seed"]) == key]
        if len(points) < 5:
            raise ValueError(f"trajectory too short for {key}: {len(points)} checkpoints")
    rows.sort(key=lambda row: (row["arm"], row["seed"], row["checkpoint_step"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "complete", "target": "twibot20", "arms": list(ARMS),
        "seeds": [0, 1, 2], "rows": len(rows),
        "protocol": "heldout_training_edges_width256_existing_features_checkpoint_saturation",
    }
    (output.parent / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

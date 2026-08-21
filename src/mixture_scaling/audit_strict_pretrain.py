from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def audit_run(run_dir: Path, expected_sources: list[str], expected_seed: int) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "validation.jsonl").read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"{run_dir.name}: no validation records")
    absolute_best = min(rows, key=lambda row: row["validation_loss"])
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    checks = {
        "status": summary.get("status") == "complete",
        "sources": summary.get("sources") == expected_sources,
        "seed": summary.get("seed") == expected_seed,
        "source_confined": summary.get("source_confined") is True,
        "split_hashes": set(summary.get("split_hashes", {})) == set(expected_sources),
        "best_step": summary.get("best_step") == absolute_best["step"] == checkpoint.get("step"),
        "best_loss": abs(summary.get("best_validation_loss", float("inf")) - absolute_best["validation_loss"]) < 1e-12,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{run_dir.name}: failed audit checks: {', '.join(failed)}")
    return {"run_id": run_dir.name, "best_step": summary["best_step"], "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pretrain-root", required=True)
    args = parser.parse_args()
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    if not manifest:
        raise ValueError("empty pretraining manifest")
    root = Path(args.pretrain_root)
    audited = [
        audit_run(root / row["run_id"], row["sources"].split(","), int(row["seed"]))
        for row in manifest
    ]
    print(json.dumps({"status": "passed", "runs": audited}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

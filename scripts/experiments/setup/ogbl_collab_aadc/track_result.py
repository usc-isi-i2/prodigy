#!/usr/bin/env python3
"""Extract AA-DC's official metrics and mirror the completed run to W&B."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


UPSTREAM = "b499c2046cfe76448545dfe08fad9effb58dd076"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_rows(text: str) -> list[dict[str, float]]:
    rows = []
    for line in text.splitlines():
        if "Official:" not in line:
            continue
        values = {key: float(value) for key, value in re.findall(r"(hits@\d+)=([0-9.eE+-]+)", line)}
        if values:
            rows.append(values)
    if len(rows) != 2:
        raise ValueError(f"expected validation and test Official rows, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    text = args.log.read_text()
    validation, test = official_rows(text)
    payload = {
        "complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "method": "AA-DC",
        "parameters": 0,
        "dataset": "ogbl-collab",
        "revision": args.revision,
        "upstream_revision": UPSTREAM,
        "raw_log": str(args.log),
        "raw_log_sha256": sha256(args.log),
        "elapsed_seconds": args.elapsed_seconds,
        "validation": validation,
        "test": test,
        "validation_edges_as_test_input": True,
        "calibration_uses_validation_only": True,
    }
    result_path = args.out / "results.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    import wandb
    run = wandb.init(
        project="ogbl-collab-aadc", group="official_b499c204", name="aadc_official_reproduction",
        mode=args.wandb_mode, dir=str(args.out),
        config={"dataset": "ogbl-collab", "method": "AA-DC", "parameters": 0,
                "revision": args.revision, "upstream_revision": UPSTREAM,
                "validation_edges_as_test_input": True},
    )
    for key, value in validation.items(): run.summary[f"validation/{key}"] = value
    for key, value in test.items(): run.summary[f"test/{key}"] = value
    run.summary["elapsed_seconds"] = args.elapsed_seconds
    url = getattr(run, "url", None)
    run.finish()
    print(json.dumps({"event": "complete", "wandb_url": url, **payload}, indent=2), flush=True)


if __name__ == "__main__":
    main()

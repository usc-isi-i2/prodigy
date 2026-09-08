#!/usr/bin/env python3
"""Build a provenance-checked external model list for the flagship ladder eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ARMS = ("baseline", "objective", "exposure", "schedule", "composition")
SEEDS = (0, 1, 2)
RUNGS = tuple(range(1, 9))
MODEL_RE = re.compile(
    r"^nmi_(?P<arm>baseline|objective|exposure|schedule|composition)_"
    r"r(?P<rung>[1-8])_s(?P<seed>[0-2])$"
)
EXPECTED = {
    (arm, rung, seed) for arm in ARMS for rung in RUNGS for seed in SEEDS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(run_dir: Path) -> list[dict]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = str(manifest["revision"])
    rows = []
    for result_path in sorted(run_dir.glob("job_*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        params = json.loads(
            (result_path.parent / "effective_config.json").read_text(encoding="utf-8")
        )
        match = MODEL_RE.fullmatch(str(params.get("prefix", "")))
        if match is None:
            continue
        key = (match["arm"], int(match["rung"]), int(match["seed"]))
        if int(params["seed"]) != key[2]:
            raise ValueError(f"training seed/prefix mismatch for {result_path}")
        state = Path(result["checkpoint_dir"]).parent
        selection_path = state / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "complete":
            raise ValueError(f"incomplete checkpoint selection: {selection_path}")
        checkpoint = Path(selection["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        rows.append(
            {
                "key": key,
                "model_id": params["prefix"],
                "checkpoint": checkpoint,
                "sources": tuple(selection["sources"]),
                "training_seed": int(params["seed"]),
                "checkpoint_step": int(selection["best_step"]),
                "training_revision": revision,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = [row for run_dir in args.run_dirs for row in load_rows(run_dir)]
    keys = [row["key"] for row in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate flagship models: {duplicates[:8]}")
    observed = set(keys)
    if observed != EXPECTED:
        missing = sorted(EXPECTED - observed)
        extra = sorted(observed - EXPECTED)
        raise ValueError(f"flagship coverage mismatch: missing={missing[:12]} extra={extra[:12]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "model_id\tcheckpoint\tsources\ttraining_seed\tcheckpoint_step\t"
        "training_revision\tcheckpoint_sha256\n"
    )
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for row in sorted(rows, key=lambda item: item["key"]):
            handle.write(
                "\t".join(
                    (
                        row["model_id"],
                        str(row["checkpoint"]),
                        ",".join(row["sources"]),
                        str(row["training_seed"]),
                        str(row["checkpoint_step"]),
                        row["training_revision"],
                        sha256(row["checkpoint"]),
                    )
                )
                + "\n"
            )
    print(f"wrote {len(rows)} verified flagship checkpoints to {args.output}")


if __name__ == "__main__":
    main()

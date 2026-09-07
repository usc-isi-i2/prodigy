#!/usr/bin/env python3
"""Resolve terminal paired checkpoints from one shared-training run."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    grouped = {"graphsage": [], "pinsage": []}
    for index, job in enumerate(manifest["jobs"]):
        encoder = "pinsage" if job["gnn_type"] == "pinsage" else "graphsage"
        candidates = sorted((args.run_dir / "state").glob(f"{job['prefix']}_*"))
        checkpoints = [p / "checkpoint" / "state_dict_2500.ckpt" for p in candidates]
        checkpoints = [p for p in checkpoints if p.is_file()]
        if len(checkpoints) != 1:
            raise RuntimeError(f"job {index} {job['prefix']}: expected one terminal checkpoint, got {checkpoints}")
        grouped[encoder].append((job["prefix"], checkpoints[0]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for encoder, rows in grouped.items():
        if len(rows) != 12:
            raise RuntimeError(f"expected twelve {encoder} checkpoints, got {len(rows)}")
        (args.out_dir / f"model_list_{encoder}.txt").write_text(
            "".join(f"{name} {path}\n" for name, path in rows)
        )


if __name__ == "__main__":
    main()

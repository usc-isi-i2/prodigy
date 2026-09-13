#!/usr/bin/env python3
"""Consolidate recovered RQ1 pretraining trajectories without collapsing retries."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


FILE_RE = re.compile(r"(metrics|scores)_(val|test)_step(\d+)\.json$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_from_run(run_name: str) -> str:
    match = re.match(r"rq1_loto_(.+)_pretrain_s[12]_\d+$", run_name)
    if not match:
        raise ValueError(f"Unrecognized run name: {run_name}")
    return match.group(1)


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed1-root", type=Path, required=True)
    parser.add_argument("--seed2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    roots = [(1, "prodigy-rq1-cache", args.seed1_root), (2, "prodigy-rq1-revised2", args.seed2_root)]
    trajectory_rows = []
    manifest_rows = []

    for seed, worktree, root in roots:
        pairs = {}
        for path in sorted(root.glob("pretrain*/**/data/*.json")):
            match = FILE_RE.fullmatch(path.name)
            if not match:
                continue
            kind, split, update = match.groups()
            attempt = path.relative_to(root).parts[0]
            run_name = path.parent.parent.name
            key = (attempt, run_name, split, int(update))
            pairs.setdefault(key, {})[kind] = path

        for (attempt, run_name, split, update), paths in sorted(pairs.items()):
            if set(paths) != {"metrics", "scores"}:
                raise ValueError(f"Unpaired metric files for {(seed, attempt, run_name, split, update)}: {paths}")
            metrics = load(paths["metrics"])
            scores = load(paths["scores"])
            accuracy = metrics[f"{split}_accuracy"]
            if not math.isclose(accuracy, scores[f"{split}_acc"], rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"Accuracy mismatch for {paths['metrics']}")
            trajectory_rows.append({
                "seed": seed,
                "source_worktree": worktree,
                "attempt": attempt,
                "target": target_from_run(run_name),
                "split": split,
                "update": update,
                "accuracy": accuracy,
                "accuracy_std": scores[f"{split}_acc_std"],
                "macro_f1": metrics[f"{split}_f1"],
                "roc_auc": metrics[f"{split}_roc_auc"],
                "loss": scores[f"{split}_loss"],
                "aux_loss": scores[f"{split}_aux_loss"],
                "metrics_sha256": digest(paths["metrics"]),
                "scores_sha256": digest(paths["scores"]),
                "metrics_source_relpath": str(paths["metrics"].relative_to(root)),
                "scores_source_relpath": str(paths["scores"].relative_to(root)),
            })

        for path in sorted(root.glob("pretrain*/manifest/*.json")):
            manifest = load(path)
            attempt = path.relative_to(root).parts[0]
            manifest_rows.append({
                "seed": seed,
                "source_worktree": worktree,
                "attempt": attempt,
                "target": manifest["target"],
                "excluded_families_json": json.dumps(manifest["excluded_family"], separators=(",", ":")),
                "sources_json": json.dumps(manifest["sources"], separators=(",", ":")),
                "checkpoint_original_path": manifest["checkpoint"],
                "manifest_sha256": digest(path),
                "manifest_source_relpath": str(path.relative_to(root)),
            })

    if len(trajectory_rows) != 436 or len(manifest_rows) != 8:
        raise ValueError(f"Unexpected recovery counts: {len(trajectory_rows)} trajectories, {len(manifest_rows)} manifests")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("pretraining_trajectories.csv", trajectory_rows), ("checkpoint_manifests.csv", manifest_rows)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the sparse 0/1/3/10-shot extension of the atomic train1 matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[6]
BASE = HERE / "data" / "atomic_train1_matrix_15x15_provenance.tsv"
WANDB = ROOT / "wandb_exports" / "graph_clip_runs.csv"
WIDE = HERE / "data" / "atomic_train1_matrix_all_shots.tsv"
LONG = HERE / "data" / "atomic_train1_matrix_all_shots_provenance.tsv"
SHOTS = (0, 1, 3, 10)


def token(value: str) -> str:
    return value.replace("+", "_")


def main() -> None:
    with BASE.open(newline="", encoding="utf-8") as handle:
        base = list(csv.DictReader(handle, delimiter="\t"))
    sources = list(dict.fromkeys(row["train_id"] for row in base))
    targets = list(dict.fromkeys(row["eval_id"] for row in base))
    checkpoints = {}
    for row in base:
        checkpoints.setdefault(row["train_id"], row["pretrained_model_run"])

    candidates = {(s, t, k): [] for s in sources for t in targets for k in SHOTS}
    with WANDB.open(newline="", encoding="utf-8") as handle:
        for run in csv.DictReader(handle):
            if run["state"] != "finished":
                continue
            try:
                params = json.loads(run["config_json"] or "{}").get("params", {})
                summary = json.loads(run["summary_json"] or "{}")
                auc = float(summary["test_roc_auc"])
                shots = int(params["n_shots"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if shots not in SHOTS:
                continue
            for source in sources:
                prefix = f"eval_{token(source)}_to_"
                if not run["run_name"].startswith(prefix):
                    continue
                if params.get("pretrained_model_run") != checkpoints[source]:
                    break
                for target in targets:
                    if run["run_name"].startswith(prefix + token(target) + "_"):
                        candidates[(source, target, shots)].append((run, params, auc))
                        break
                break

    base3 = {(r["train_id"], r["eval_id"]): r for r in base}
    selected = {}
    for key, choices in candidates.items():
        source, target, shots = key
        if shots == 3:
            run_id = base3[(source, target)]["run_id"]
            exact = [choice for choice in choices if choice[0]["run_id"] == run_id]
            if exact:
                selected[key] = exact[0]
                continue
        if choices:
            selected[key] = max(choices, key=lambda choice: choice[0]["created_at"])

    columns = [f"{target}|{shots}shot" for target in targets for shots in SHOTS]
    with WIDE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_id", *columns], delimiter="\t")
        writer.writeheader()
        for source in sources:
            row = {"train_id": source}
            for target in targets:
                for shots in SHOTS:
                    choice = selected.get((source, target, shots))
                    row[f"{target}|{shots}shot"] = "" if choice is None else f"{choice[2]:.10f}"
            writer.writerow(row)

    fields = ["train_id", "eval_id", "shots", "roc_auc", "result_status", "run_id", "run_name", "created_at"]
    with LONG.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for source in sources:
            for target in targets:
                for shots in SHOTS:
                    choice = selected.get((source, target, shots))
                    if choice is None:
                        writer.writerow({"train_id": source, "eval_id": target, "shots": shots, "result_status": "missing"})
                    else:
                        run, _, auc = choice
                        writer.writerow({
                            "train_id": source, "eval_id": target, "shots": shots,
                            "roc_auc": f"{auc:.10f}",
                            "result_status": "void_pre_20260723" if target.endswith("+lp") else "reported_legacy",
                            "run_id": run["run_id"], "run_name": run["run_name"], "created_at": run["created_at"],
                        })
    print(f"wrote {len(sources)}x{len(columns)} sparse matrix; observed={len(selected)} missing={len(candidates)-len(selected)}")


if __name__ == "__main__":
    main()

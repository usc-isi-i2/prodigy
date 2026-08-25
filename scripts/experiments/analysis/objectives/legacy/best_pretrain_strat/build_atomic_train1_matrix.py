#!/usr/bin/env python3
"""Recover the atomic train1 graph+task evaluation matrix from W&B exports.

The historical June table is authoritative for its eleven atomic rows.  Four
later small-graph rows (Election 2020 and suspended Ukraine, NM and PL) are
recovered from the final 3-shot W&B runs that include the anchor cell
``eval_election2020_nm_to_covid_political_nm_3shot_...``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[6]
LEGACY = HERE / "data" / "experiment_results_Jun_25.tsv"
WANDB = ROOT / "wandb_exports" / "graph_clip_runs.csv"
WIDE_OUT = HERE / "data" / "atomic_train1_matrix_15x15.tsv"
LONG_OUT = HERE / "data" / "atomic_train1_matrix_15x15_provenance.tsv"

RECOVERED_SOURCES = [
    "covid_political+nm",
    "covid_political+pl",
    "election2020+nm",
    "election2020+pl",
    "ukr_rus_suspended+nm",
    "ukr_rus_suspended+pl",
]


def run_token(value: str) -> str:
    return value.replace("+", "_")


def load_runs(sources: list[str], targets: list[str]) -> dict[tuple[str, str], list[dict]]:
    found: dict[tuple[str, str], list[dict]] = {(s, t): [] for s in sources for t in targets}
    with WANDB.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["state"] != "finished" or "_3shot_" not in row["run_name"]:
                continue
            try:
                summary = json.loads(row["summary_json"] or "{}")
                params = json.loads(row["config_json"] or "{}").get("params", {})
                auc = float(summary["test_roc_auc"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            for source in sources:
                prefix = f"eval_{run_token(source)}_to_"
                if not row["run_name"].startswith(prefix):
                    continue
                for target in targets:
                    if row["run_name"].startswith(prefix + run_token(target) + "_"):
                        found[(source, target)].append({**row, "auc": auc, "params": params})
                        break
                break
    return found


def main() -> None:
    with LEGACY.open(newline="", encoding="utf-8") as handle:
        legacy_rows = list(csv.DictReader(handle, delimiter="\t"))
    legacy_targets = [field for field in legacy_rows[0] if field != "train_id"]
    targets = []
    for graph in ("covid19_twitter", "covid_political", "election2020", "midterm", "ukr_rus_suspended", "ukr_rus_twitter"):
        tasks = ("lp", "nm", "pl") if graph in {"covid19_twitter", "midterm", "ukr_rus_twitter"} else ("nm", "pl")
        targets.extend(f"{graph}+{task}" for task in tasks)
    atomic = [row for row in legacy_rows if ">" not in row["train_id"] and "/" not in row["train_id"]]
    old_sources = [row["train_id"] for row in atomic]
    sources = old_sources + RECOVERED_SOURCES
    candidates = load_runs(sources, targets)

    selected: dict[tuple[str, str], dict] = {}
    values: dict[tuple[str, str], float] = {}
    legacy_by_source = {row["train_id"]: row for row in atomic}

    for source in old_sources:
        for target in legacy_targets:
            expected = float(legacy_by_source[source][target])
            choices = candidates[(source, target)]
            if not choices:
                raise RuntimeError(f"No W&B candidate for {source} -> {target}")
            choice = min(choices, key=lambda item: (abs(item["auc"] - expected), item["created_at"]))
            if abs(choice["auc"] - expected) > 1e-9:
                raise RuntimeError(
                    f"Could not provenance-match {source} -> {target}: "
                    f"TSV={expected}, closest W&B={choice['auc']}"
                )
            selected[(source, target)] = choice
            values[(source, target)] = expected

    for source in sources:
        source_token = run_token(source)
        for target in targets:
            if (source, target) in selected:
                continue
            choices = [
                item for item in candidates[(source, target)]
                if f"train1_{source_token}_" in item["params"].get("pretrained_model_run", "")
            ]
            if not choices:
                raise RuntimeError(f"No final train1 candidate for {source} -> {target}")
            choice = max(choices, key=lambda item: item["created_at"])
            selected[(source, target)] = choice
            values[(source, target)] = choice["auc"]

    with WIDE_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_id", *targets], delimiter="\t")
        writer.writeheader()
        for source in sources:
            writer.writerow({
                "train_id": source,
                **{target: f"{values[(source, target)]:.10f}" for target in targets},
            })

    fields = [
        "train_id", "eval_id", "roc_auc", "result_status", "run_id", "run_name",
        "created_at", "pretrained_model_run", "n_shots", "n_way", "n_query",
        "eval_random_query", "test_len_cap", "val_len_cap", "task_name", "n_hop",
    ]
    with LONG_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for source in sources:
            for target in targets:
                item = selected[(source, target)]
                params = item["params"]
                writer.writerow({
                    "train_id": source,
                    "eval_id": target,
                    "roc_auc": f"{values[(source, target)]:.10f}",
                    "result_status": "void_pre_20260723" if target.endswith("+lp") else "reported_legacy",
                    "run_id": item["run_id"],
                    "run_name": item["run_name"],
                    "created_at": item["created_at"],
                    "pretrained_model_run": params.get("pretrained_model_run", ""),
                    "n_shots": params.get("n_shots", ""),
                    "n_way": params.get("n_way", ""),
                    "n_query": params.get("n_query", ""),
                    "eval_random_query": params.get("eval_random_query", ""),
                    "test_len_cap": params.get("test_len_cap", ""),
                    "val_len_cap": params.get("val_len_cap", ""),
                    "task_name": params.get("task_name", ""),
                    "n_hop": params.get("n_hop", ""),
                })

    print(f"wrote {len(sources)}x{len(targets)} matrix to {WIDE_OUT}")
    print(f"wrote {len(sources) * len(targets)} provenance rows to {LONG_OUT}")


if __name__ == "__main__":
    main()

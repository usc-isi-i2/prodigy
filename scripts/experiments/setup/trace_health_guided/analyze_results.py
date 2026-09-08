#!/usr/bin/env python3
"""Summarize the matched-budget health-guided comparison on the fresh stream."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from scripts.experiments.setup.trace_health_guided.make_stage1_plan import SOURCES


FIXED_METHODS = {
    "blocked_sequential": "r4_blocked_s{seed}",
    "replay100": "r4_replay100_s{seed}",
    "uniform_interleaving": "r4_interleaved_s{seed}",
    "naive_merged": "merged_s{seed}",
    "health_guided": "health_guided_s{seed}",
}


def load_metrics(directory: Path) -> tuple[dict[str, dict], str]:
    rows = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected = {
        row["model_id"]: row
        for row in rows
        if row["variant"] == "baseline" and row["decoder"] == "full_model"
    }
    fingerprints = {row["episode_fingerprint"] for row in selected.values()}
    if len(fingerprints) != 1:
        raise ValueError(f"models did not share one episode stream in {directory}")
    return selected, next(iter(fingerprints))


def load_health_rankings(path: Path) -> dict[int, tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["seed"]): tuple(row["ranked_sources"].split(","))
            for row in csv.DictReader(handle, delimiter="\t")
        }


def cell(
    *, seed: int, method: str, model_id: str, row: dict,
    selection: str, uses_query_labels: bool,
) -> dict:
    return {
        "seed": seed,
        "method": method,
        "model_id": model_id,
        "selection": selection,
        "uses_query_labels_for_selection": uses_query_labels,
        "accuracy": row["accuracy"],
        "roc_auc": row["roc_auc"],
        "nll": row["nll"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-directory", type=Path, required=True)
    parser.add_argument("--fresh-directory", type=Path, required=True)
    parser.add_argument("--health-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, original_fingerprint = load_metrics(args.original_directory)
    fresh, fresh_fingerprint = load_metrics(args.fresh_directory)
    if set(original) != set(fresh):
        raise ValueError("original and fresh model lattices differ")
    if original_fingerprint == fresh_fingerprint:
        raise ValueError("original and fresh streams are not disjoint")
    rankings = load_health_rankings(args.health_plan)
    cells = []
    for seed in range(3):
        for method, template in FIXED_METHODS.items():
            model_id = template.format(seed=seed)
            cells.append(
                cell(
                    seed=seed,
                    method=method,
                    model_id=model_id,
                    row=fresh[model_id],
                    selection="fixed_preregistered",
                    uses_query_labels=False,
                )
            )
        specialist_ids = [f"ss_{source}_s{seed}" for source in SOURCES]
        health_source = rankings[seed][0]
        health_single = f"ss_{health_source}_s{seed}"
        cells.append(
            cell(
                seed=seed,
                method="health_selected_single",
                model_id=health_single,
                row=fresh[health_single],
                selection="original_unlabeled_u1_agreement",
                uses_query_labels=False,
            )
        )
        validation_best = max(specialist_ids, key=lambda model_id: original[model_id]["roc_auc"])
        cells.append(
            cell(
                seed=seed,
                method="validation_selected_single",
                model_id=validation_best,
                row=fresh[validation_best],
                selection="original_query_auc",
                uses_query_labels=True,
            )
        )
        oracle_best = max(specialist_ids, key=lambda model_id: fresh[model_id]["roc_auc"])
        cells.append(
            cell(
                seed=seed,
                method="oracle_best_single",
                model_id=oracle_best,
                row=fresh[oracle_best],
                selection="fresh_query_auc_oracle",
                uses_query_labels=True,
            )
        )
    methods = sorted({row["method"] for row in cells})
    summary = []
    for method in methods:
        group = [row for row in cells if row["method"] == method]
        summary.append(
            {
                "method": method,
                "seeds": len(group),
                "mean_accuracy": mean(row["accuracy"] for row in group),
                "std_accuracy": stdev(row["accuracy"] for row in group),
                "mean_roc_auc": mean(row["roc_auc"] for row in group),
                "std_roc_auc": stdev(row["roc_auc"] for row in group),
                "mean_nll": mean(row["nll"] for row in group),
                "uses_query_labels_for_selection": any(
                    row["uses_query_labels_for_selection"] for row in group
                ),
            }
        )
    uniform = {
        row["seed"]: row for row in cells if row["method"] == "uniform_interleaving"
    }
    deltas = []
    for row in cells:
        reference = uniform[row["seed"]]
        deltas.append(
            {
                "seed": row["seed"],
                "method": row["method"],
                "delta_accuracy_vs_uniform": row["accuracy"] - reference["accuracy"],
                "delta_roc_auc_vs_uniform": row["roc_auc"] - reference["roc_auc"],
                "delta_nll_vs_uniform": row["nll"] - reference["nll"],
            }
        )
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (
        ("comparison_cells.csv", cells),
        ("comparison_summary.csv", summary),
        ("deltas_vs_uniform.csv", deltas),
    ):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "target": "facebook_page_reference",
                "training_updates_per_model": 2500,
                "training_sources": list(SOURCES),
                "original_episode_fingerprint": original_fingerprint,
                "fresh_episode_fingerprint": fresh_fingerprint,
                "primary_method": "health_guided",
                "primary_baseline": "uniform_interleaving",
                "selection_boundary": (
                    "Health-guided scheduling uses original target inputs and episode "
                    "support labels, but no target query labels."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Matched-budget fresh-stream summary")
    for row in sorted(summary, key=lambda value: value["mean_roc_auc"], reverse=True):
        print(
            f"{row['method']:28s} acc={row['mean_accuracy']:.4f} "
            f"auc={row['mean_roc_auc']:.4f} nll={row['mean_nll']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

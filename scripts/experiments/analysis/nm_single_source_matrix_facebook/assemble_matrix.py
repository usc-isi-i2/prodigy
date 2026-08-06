#!/usr/bin/env python3
"""Build the 9x9 NM matrix from the committed 8x8 matrix and Facebook cells."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "nm_single_source_matrix" / "data"
DATA = HERE / "data"
GRAPHS = [
    "ukr_rus_twitter",
    "covid19_twitter",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk_twitter",
    "facebook_page_reference",
]
FACEBOOK = GRAPHS[-1]
METRICS = ["roc_auc", "accuracy", "f1"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_long = read_csv(BASE / "nm_single_source_matrix_long.csv")
    extension = read_csv(DATA / "facebook_extension_metrics.csv")

    expected_extension = {(source, FACEBOOK) for source in GRAPHS[:-1]}
    expected_extension |= {(FACEBOOK, target) for target in GRAPHS}
    actual_extension = {(row["train_graph"], row["test_graph"]) for row in extension}
    if len(extension) != 17 or actual_extension != expected_extension:
        missing = sorted(expected_extension - actual_extension)
        extra = sorted(actual_extension - expected_extension)
        raise ValueError(f"expected 17 Facebook cells; missing={missing}, extra={extra}")

    auc: dict[tuple[str, str], float] = {}
    for row in base_long:
        if row["metric"] == "roc_auc":
            auc[row["train"], row["test"]] = float(row["value"])
    for row in extension:
        auc[row["train_graph"], row["test_graph"]] = float(row["roc_auc"])
    expected_matrix = {(source, target) for source in GRAPHS for target in GRAPHS}
    if set(auc) != expected_matrix:
        raise ValueError("assembled AUC matrix is incomplete")

    wide_rows: list[dict[str, object]] = []
    for source in GRAPHS:
        row: dict[str, object] = {"train_graph": source}
        row.update({target: f"{auc[source, target]:.6f}" for target in GRAPHS})
        wide_rows.append(row)
    write_csv(DATA / "nm_single_source_matrix_9x9.csv", ["train_graph", *GRAPHS], wide_rows)

    values: dict[tuple[str, str, str], float] = {}
    for row in base_long:
        values[row["metric"], row["train"], row["test"]] = float(row["value"])
    for row in extension:
        for metric in METRICS:
            values[metric, row["train_graph"], row["test_graph"]] = float(row[metric])
    expected_long = {
        (metric, source, target)
        for metric in METRICS
        for source in GRAPHS
        for target in GRAPHS
    }
    if set(values) != expected_long:
        raise ValueError("assembled long-form matrix is incomplete")
    long_rows = [
        {"metric": metric, "train": source, "test": target, "value": f"{values[metric, source, target]:.12f}"}
        for metric in METRICS
        for source in GRAPHS
        for target in GRAPHS
    ]
    write_csv(DATA / "nm_single_source_matrix_9x9_long.csv", ["metric", "train", "test", "value"], long_rows)

    incoming = sorted(
        ((source, auc[source, FACEBOOK]) for source in GRAPHS[:-1]),
        key=lambda item: item[1],
        reverse=True,
    )
    facebook_to_twitter = {target: auc[FACEBOOK, target] for target in GRAPHS[:-1]}
    foreign_ranks: dict[str, int] = {}
    best_foreign_by_target: dict[str, dict[str, object]] = {}
    for target in GRAPHS[:-1]:
        foreign = sorted(
            ((source, auc[source, target]) for source in GRAPHS if source != target),
            key=lambda item: item[1],
            reverse=True,
        )
        foreign_ranks[target] = next(index for index, item in enumerate(foreign, 1) if item[0] == FACEBOOK)
        best_foreign_by_target[target] = {"source": foreign[0][0], "roc_auc": foreign[0][1]}

    asymmetry = {
        graph: auc[graph, FACEBOOK] - auc[FACEBOOK, graph]
        for graph in GRAPHS[:-1]
    }
    donor_means = {
        source: mean(auc[source, target] for target in GRAPHS if target != source)
        for source in GRAPHS
    }
    receiver_means = {
        target: mean(auc[source, target] for source in GRAPHS if source != target)
        for target in GRAPHS
    }
    facebook_gaps = {
        target: auc[FACEBOOK, target]
        - max(auc[source, target] for source in GRAPHS[:-1] if source != target)
        for target in GRAPHS[:-1]
    }
    summary = {
        "protocol": {
            "checkpoint_step": 40000,
            "objective": "neighbor_matching",
            "n_hop": 1,
            "n_way": 30,
            "n_shots": 3,
            "eval_episodes": 500,
            "seeds": 1,
        },
        "facebook_diagonal_roc_auc": auc[FACEBOOK, FACEBOOK],
        "best_foreign_to_facebook": {"source": incoming[0][0], "roc_auc": incoming[0][1]},
        "facebook_diagonal_gain_over_best_foreign": auc[FACEBOOK, FACEBOOK] - incoming[0][1],
        "mean_foreign_to_facebook_roc_auc": mean(value for _, value in incoming),
        "facebook_to_twitter_mean_roc_auc": mean(facebook_to_twitter.values()),
        "facebook_to_twitter_best": max(facebook_to_twitter, key=facebook_to_twitter.get),
        "facebook_to_twitter_worst": min(facebook_to_twitter, key=facebook_to_twitter.get),
        "facebook_foreign_donor_ranks": foreign_ranks,
        "facebook_mean_foreign_donor_rank": mean(foreign_ranks.values()),
        "facebook_gap_to_best_twitter_foreign_donor": facebook_gaps,
        "best_foreign_by_twitter_target": best_foreign_by_target,
        "off_diagonal_donor_mean_roc_auc": donor_means,
        "off_diagonal_receiver_mean_roc_auc": receiver_means,
        "twitter_to_facebook_minus_facebook_to_twitter": asymmetry,
        "mean_transfer_asymmetry": mean(asymmetry.values()),
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

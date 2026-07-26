#!/usr/bin/env python3
"""Pivot the long feature-ablation results into a wide, one-row-per-config sheet.

Reads feature_ablation_results.csv (long: config, condition, accuracy, roc_auc)
and writes feature_ablation_wide.csv with one row per config and all four
conditions spread across columns, plus parsed metadata (train/test task, graph).

Usage:
  python3 build_results_wide.py \
    --in feature_ablation_results.csv --out feature_ablation_wide.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

GRAPH_MAP = {  # test_graph label (matches the hand-made sheet: covid* -> covid)
    "covid19_twitter": "covid",
    "covid_political": "covid",
    "midterm": "midterm",
    "twibot20": "twibot20",
    "election2020": "election2020",
}
TRAIN_TASK = "nm"  # checkpoint is nm_matrix_covid


def parse_config(cfg: str) -> tuple[str, str]:
    """-> (test_task, test_graph) from '...to_<graph>_<task>_...'."""
    post = cfg.split("_to_", 1)[1]
    for tok, task in (("_nm_", "nm"), ("_pl_", "pl")):
        if tok in post:
            graph = post.split(tok, 1)[0]
            return task, GRAPH_MAP.get(graph, graph)
    return "?", "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="feature_ablation_results.csv")
    ap.add_argument("--out", default="feature_ablation_wide.csv")
    ap.add_argument("--drop", default="_3way",
                    help="Skip configs whose key contains this substring (default: the 3-way smoke).")
    args = ap.parse_args()

    # config -> condition -> (acc, auc)
    data: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    order: list[str] = []
    with open(args.inp, newline="") as f:
        for r in csv.DictReader(f):
            cfg = r["config"]
            if args.drop and args.drop in cfg:
                continue
            if cfg not in data:
                order.append(cfg)
            data[cfg][r["condition"]] = (r["accuracy"], r["roc_auc"])

    # group nm rows before pl (matches the hand-made sheet), graph alphabetical within
    order.sort(key=lambda c: (parse_config(c)[0] != "nm", c))

    fields = ["config", "condition", "accuracy", "auc", "control_acc", "control_auc",
              "train_task", "test_task", "test_graph",
              "acc_intact", "acc_zero", "acc_permute", "acc_noise",
              "auc_intact", "auc_zero", "auc_permute", "auc_noise", "keep", "task-graph"]
    conds = ["intact", "zero", "permute", "noise"]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cfg in order:
            by = data[cfg]
            acc = {c: by.get(c, ("", ""))[0] for c in conds}
            auc = {c: by.get(c, ("", ""))[1] for c in conds}
            test_task, test_graph = parse_config(cfg)
            w.writerow({
                "config": cfg, "condition": "intact",
                "accuracy": acc["intact"], "auc": auc["intact"],
                "control_acc": acc["intact"], "control_auc": auc["intact"],
                "train_task": TRAIN_TASK, "test_task": test_task, "test_graph": test_graph,
                "acc_intact": acc["intact"], "acc_zero": acc["zero"],
                "acc_permute": acc["permute"], "acc_noise": acc["noise"],
                "auc_intact": auc["intact"], "auc_zero": auc["zero"],
                "auc_permute": auc["permute"], "auc_noise": auc["noise"],
                "keep": 1, "task-graph": f"{test_graph}+{test_task}",
            })
    print(f"wrote {args.out} ({len(order)} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

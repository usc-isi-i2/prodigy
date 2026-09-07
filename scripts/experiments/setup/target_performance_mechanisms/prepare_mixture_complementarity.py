"""Snapshot the complete historical CLS lattice; keep role-corrected runs separate."""
import argparse
import ast
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path

import pandas as pd

from scripts.experiments.setup.final_core.core_plan import SOURCES

TARGETS = {"covid_political", "election2020", "facebook_page_reference", "twibot20", "ukr_rus_suspended"}


def validate_lattice(cells, models):
    if len(cells) != 270 or cells.duplicated(["model_id", "dataset"]).any() or set(cells.dataset) != TARGETS:
        raise ValueError("requires complete 54-model by five-target lattice")
    for key, value in {"checkpoint_step": 2500, "training_seed": 0, "evaluation_seed": 0,
                       "eval_episode_seed_offset": 0, "episodes": 128, "n_way": 2, "n_shot": 10,
                       "architecture": "prodigy", "task": "classification"}.items():
        if not cells[key].eq(value).all():
            raise ValueError(f"unexpected historical protocol: {key}")
    if len(models) != 54 or models.model_id.duplicated().any() or models.checkpoint.duplicated().any():
        raise ValueError("requires unique historical models and checkpoints")
    cells = cells.copy()
    models = models.copy()
    cells["source_set"] = cells.sources.map(lambda s: tuple(sorted(ast.literal_eval(s))))
    models["source_set"] = models.sources.map(lambda s: tuple(sorted(s.split(","))))
    expected = {tuple(c) for k in (1, 2, 8) for c in combinations(sorted(SOURCES), k)}
    if models.source_set.duplicated().any() or set(models.source_set) != expected or set(cells.model_id) != set(models.model_id):
        raise ValueError("incomplete singleton/pair/LOO compositions")
    lookup = models.set_index("model_id").source_set
    if not (cells.source_set == cells.model_id.map(lookup)).all():
        raise ValueError("aggregate and checkpoint compositions disagree")
    for target, group in cells.groupby("dataset"):
        if group.episode_fingerprint.nunique() != 1 or group.queries.nunique() != 1 or not group.queries.eq(group.episodes * group.n_way * group.n_query).all():
            raise ValueError(f"inconsistent cached target protocol: {target}")
    for metric in ("roc_auc", "accuracy", "f1"):
        if not cells[metric].between(0, 1).all():
            raise ValueError("invalid reference metric")
    return models


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = pd.read_csv(args.metrics, sep="\t")
    models = validate_lattice(cells, pd.read_csv(args.models, sep="\t"))
    if args.output.exists():
        raise ValueError("refusing to replace a prior frozen input snapshot")
    args.output.mkdir(parents=True)
    cells.to_csv(args.output / "classification_long.tsv", sep="\t", index=False)
    models.drop(columns="source_set").to_csv(args.output / "model_list.tsv", sep="\t", index=False)
    mixtures = models[models.source_set.map(len) > 1]
    mixtures.drop(columns="source_set").to_csv(args.output / "mixture_model_list.tsv", sep="\t", index=False)
    details = [{"model_id": r.model_id, "checkpoint": r.checkpoint, "sources": list(r.source_set),
                "source_count": len(r.source_set), "checkpoint_step": 2500, "training_seed": 0}
               for r in models.itertuples()]
    (args.output / "manifest.json").write_text(json.dumps({"models": details, "aggregate_cells": 270,
        "mixtures": 45, "new_training": False, "role_corrected_runs_included": False,
        "input_metrics": str(args.metrics.resolve()), "input_models": str(args.models.resolve()),
        "input_metrics_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
        "input_models_sha256": hashlib.sha256(args.models.read_bytes()).hexdigest(),
        "snapshot_metrics_sha256": hashlib.sha256((args.output / "classification_long.tsv").read_bytes()).hexdigest(),
        "snapshot_models_sha256": hashlib.sha256((args.output / "model_list.tsv").read_bytes()).hexdigest(),
        "ensemble_rule": "equal constituent probability", "secondary_ensemble_rule": "equal constituent logit",
        "test_label_fitting": False, "exploratory_after_aggregate_results": True}, indent=2) + "\n")
    print(json.dumps({"models": len(models), "mixture_models": len(mixtures), "aggregate_cells": len(cells)}))


if __name__ == "__main__":
    main()

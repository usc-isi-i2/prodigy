from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .probe_strict import classifier
from .strict_data import induced_partition, load_raw, load_split, split_hash
from .strict_metrics import classification_metrics


def labeled_features(graph) -> tuple[np.ndarray, np.ndarray]:
    features = graph.data.x.numpy()
    labels = graph.data.y.numpy()
    mask = labels >= 0
    return features[mask], labels[mask]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--target", default="twibot20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    config = load_config(args.config)
    raw = load_raw(config["graphs"][args.target]["path"])
    split = load_split(Path(args.split_root) / f"{args.target}.pt", args.target, int(raw["x"].shape[0]))

    train_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "train")
    validation_graph = induced_partition(
        args.target, config["graphs"][args.target]["path"], split, "validation"
    )
    train_x, train_y = labeled_features(train_graph)
    validation_x, validation_y = labeled_features(validation_graph)
    trials = []
    for c_value in map(float, config["protocol"]["probe_c_values"]):
        head = classifier(c_value, args.seed)
        head.fit(train_x, train_y)
        metrics = classification_metrics(
            validation_y, head.predict(validation_x), head.predict_proba(validation_x), head.classes_
        )
        trials.append({"c": c_value, **metrics})
    chosen = max(trials, key=lambda row: (row["f1_macro"], row["roc_auc_ovr_macro"], -row["c"]))
    final_head = classifier(float(chosen["c"]), args.seed)
    final_head.fit(np.concatenate((train_x, validation_x)), np.concatenate((train_y, validation_y)))

    # Construct and score the test partition only after hyperparameter selection.
    test_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "test")
    test_x, test_y = labeled_features(test_graph)
    result = {
        "status": "complete",
        "evaluation": "raw_feature_logistic_regression",
        "target": args.target,
        "target_split_hash": split_hash(split),
        "seed": args.seed,
        "selected_c": chosen["c"],
        "selection_metric": "validation_f1_macro",
        "validation_trials": trials,
        "train_nodes": int(len(train_y)),
        "validation_nodes": int(len(validation_y)),
        "test_nodes": int(len(test_y)),
        **classification_metrics(
            test_y, final_head.predict(test_x), final_head.predict_proba(test_x), final_head.classes_
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

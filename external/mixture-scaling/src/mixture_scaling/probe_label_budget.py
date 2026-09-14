from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mixture_scaling.config import load_config
from mixture_scaling.model import GraphSAGE
from mixture_scaling.probe_strict import choose_trial, classifier, embed
from mixture_scaling.strict_data import induced_partition, load_raw, load_split, split_hash
from mixture_scaling.strict_metrics import classification_metrics
from mixture_scaling.structural_logistic_baseline import structural_features

BUDGETS = (1, 5, 10, 20, 50)


def structuralize(graph, mean=None, std=None):
    values, names = structural_features(graph.data.edge_index, int(graph.data.num_nodes))
    if mean is None:
        mean = values.mean(axis=0, dtype=np.float64)
        std = values.std(axis=0, dtype=np.float64)
        std[std < 1e-8] = 1.0
    graph.data.x = torch.from_numpy(((values - mean) / std).astype(np.float32))
    return names, mean, std


def support_indices(labels: np.ndarray, budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 104729 * budget)
    chosen = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < budget:
            raise ValueError(f"class {label} has {len(candidates)} training nodes; budget={budget}")
        chosen.extend(rng.choice(candidates, size=budget, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.device not in (2, 3):
        raise ValueError("label-budget probe may use only GPUs 2 and 3")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    if int(metadata["seed"]) != args.seed:
        raise ValueError("checkpoint seed mismatch")
    config = load_config(args.config)
    protocol = dict(config["protocol"])
    protocol.update(metadata["protocol"])
    raw = load_raw(config["graphs"][args.target]["path"])
    split = load_split(Path(args.split_root) / f"{args.target}.pt", args.target, int(raw["x"].shape[0]))
    train_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "train")
    validation_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "validation")
    structural_names = None
    if metadata["feature_mode"] == "structural":
        structural_names, mean, std = structuralize(train_graph)
        structuralize(validation_graph, mean, std)
    model = GraphSAGE(int(protocol["input_dim"]), int(protocol["hidden_dim"]),
                      int(protocol["output_dim"]), int(protocol["layers"]),
                      float(protocol["dropout"]))
    model.load_state_dict(checkpoint["model"])
    device = torch.device(f"cuda:{args.device}")
    model.to(device)
    train_x, train_y = embed(model, train_graph, device)
    validation_x, validation_y = embed(model, validation_graph, device)
    selections = {}
    for budget in BUDGETS:
        support = support_indices(train_y, budget, args.seed)
        trials = []
        for c_value in map(float, protocol["probe_c_values"]):
            head = classifier(c_value, args.seed)
            head.fit(train_x[support], train_y[support])
            trials.append({"c": c_value, **classification_metrics(
                validation_y, head.predict(validation_x), head.predict_proba(validation_x), head.classes_)})
        selections[budget] = (support, choose_trial(trials, "auc"), trials)
    test_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "test")
    if metadata["feature_mode"] == "structural":
        structuralize(test_graph, mean, std)
    test_x, test_y = embed(model, test_graph, device)
    results = []
    for budget in BUDGETS:
        support, chosen, trials = selections[budget]
        head = classifier(float(chosen["c"]), args.seed)
        head.fit(train_x[support], train_y[support])
        results.append({"labels_per_class": budget, "support_nodes": int(len(support)),
                        "selected_c": chosen["c"], "validation_trials": trials,
                        **classification_metrics(test_y, head.predict(test_x),
                                                 head.predict_proba(test_x), head.classes_)})
    result = {"status": "complete", "evaluation": "frozen_low_label_probe",
              "target": args.target, "condition": args.condition,
              "target_split_hash": split_hash(split), "checkpoint": args.checkpoint,
              "checkpoint_step": int(checkpoint["step"]), "sources": metadata["sources"],
              "objective": metadata["objective"], "feature_mode": metadata["feature_mode"],
              "pretrain_seed": int(metadata["seed"]), "probe_seed": args.seed,
              "budgets": list(BUDGETS), "validation_labels_used_for_fitting": False,
              "structural_feature_names": structural_names, "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

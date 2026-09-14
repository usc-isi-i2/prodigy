from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .model import GraphSAGE
from .strict_data import induced_partition, load_raw, load_split, split_hash
from .strict_metrics import classification_metrics


@torch.no_grad()
def embed(model: GraphSAGE, graph, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embedding = model(graph.data.x.to(device), graph.data.edge_index.to(device)).cpu().numpy()
    y = graph.data.y.numpy()
    labeled = y >= 0
    return embedding[labeled], y[labeled]


def classifier(c_value: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=3000, random_state=seed),
    )


def choose_trial(trials: list[dict], metric: str) -> dict:
    if metric == "auc":
        return max(trials, key=lambda row: (row["roc_auc_ovr_macro"], row["f1_macro"], -row["c"]))
    if metric == "f1":
        return max(trials, key=lambda row: (row["f1_macro"], row["roc_auc_ovr_macro"], -row["c"]))
    raise ValueError(f"unknown selection metric: {metric}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="best.pt or the literal scratch")
    parser.add_argument("--target", default="twibot20")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection-metric", choices=("auc", "f1"), default="f1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.device not in (2, 3):
        raise ValueError("strict pilot may use only GPUs 2 and 3")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    config = load_config(args.config)
    protocol = dict(config["protocol"])
    raw = load_raw(config["graphs"][args.target]["path"])
    split = load_split(Path(args.split_root) / f"{args.target}.pt", args.target, int(raw["x"].shape[0]))
    graphs = {
        partition: induced_partition(args.target, config["graphs"][args.target]["path"], split, partition)
        for partition in ("train", "validation")
    }
    torch.manual_seed(args.seed)
    if args.checkpoint == "scratch":
        checkpoint_step, sources, pretrain_seed = 0, [], args.seed
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        protocol.update(checkpoint["metadata"].get("protocol", {}))
        checkpoint_step = int(checkpoint["step"])
        sources = checkpoint["metadata"]["sources"]
        pretrain_seed = int(checkpoint["metadata"]["seed"])
    model = GraphSAGE(
        int(protocol["input_dim"]), int(protocol["hidden_dim"]), int(protocol["output_dim"]),
        int(protocol["layers"]), float(protocol["dropout"]),
    )
    if args.checkpoint != "scratch":
        model.load_state_dict(checkpoint["model"])
    device = torch.device(f"cuda:{args.device}")
    model.to(device)
    embedded = {partition: embed(model, graph, device) for partition, graph in graphs.items()}
    train_x, train_y = embedded["train"]
    validation_x, validation_y = embedded["validation"]
    trials = []
    for c_value in map(float, protocol["probe_c_values"]):
        head = classifier(c_value, args.seed)
        head.fit(train_x, train_y)
        prediction = head.predict(validation_x)
        probability = head.predict_proba(validation_x)
        metrics = classification_metrics(validation_y, prediction, probability, head.classes_)
        trials.append({"c": c_value, **metrics})
    chosen = choose_trial(trials, args.selection_metric)
    final_head = classifier(float(chosen["c"]), args.seed)
    final_head.fit(np.concatenate((train_x, validation_x)), np.concatenate((train_y, validation_y)))
    # The test partition is constructed, embedded, and scored only after all
    # model and hyperparameter selection is complete.
    test_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "test")
    test_x, test_y = embed(model, test_graph, device)
    test_prediction = final_head.predict(test_x)
    test_probability = final_head.predict_proba(test_x)
    test_metrics = classification_metrics(test_y, test_prediction, test_probability, final_head.classes_)
    result = {
        "status": "complete",
        "evaluation": "frozen_linear_probe",
        "target": args.target,
        "target_split_hash": split_hash(split),
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint_step,
        "sources": sources,
        "pretrain_seed": pretrain_seed,
        "probe_seed": args.seed,
        "selected_c": chosen["c"],
        "selection_metric": f"validation_{'roc_auc_ovr_macro' if args.selection_metric == 'auc' else 'f1_macro'}",
        "validation_trials": trials,
        "train_nodes": int(len(train_y)),
        "validation_nodes": int(len(validation_y)),
        "test_nodes": int(len(test_y)),
        "classes": int(len(np.unique(train_y))),
        **test_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

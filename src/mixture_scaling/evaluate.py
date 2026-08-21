from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from .config import load_config
from .data import labeled_nodes, load_graph
from .model import GraphSAGE


def fixed_split(y: np.ndarray, labels_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    support, test = [], []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        if len(indices) <= labels_per_class:
            raise ValueError(f"class {label} has only {len(indices)} labeled nodes")
        support.extend(indices[:labels_per_class])
        test.extend(indices[labels_per_class:])
    return np.asarray(support), np.asarray(test)


@torch.no_grad()
def embed(model: GraphSAGE, graph, device: torch.device) -> np.ndarray:
    model.eval()
    x = graph.data.x.to(device)
    edge_index = graph.data.edge_index.to(device)
    return model(x, edge_index).cpu().numpy()


def evaluate_embeddings(embedding: np.ndarray, y: np.ndarray, labels_per_class: int, seed: int) -> dict:
    support, test = fixed_split(y, labels_per_class, seed)
    classifier = LogisticRegression(max_iter=2000, random_state=seed)
    classifier.fit(embedding[support], y[support])
    prediction = classifier.predict(embedding[test])
    probability = classifier.predict_proba(embedding[test])
    classes = classifier.classes_
    if len(classes) == 2:
        auc = roc_auc_score(y[test], probability[:, 1], labels=classes)
    else:
        binary = label_binarize(y[test], classes=classes)
        auc = roc_auc_score(binary, probability, average="macro", multi_class="ovr")
    return {
        "roc_auc_ovr_macro": float(auc),
        "f1_macro": float(f1_score(y[test], prediction, average="macro")),
        "accuracy": float(accuracy_score(y[test], prediction)),
        "support_nodes": int(len(support)), "test_nodes": int(len(test)),
        "classes": int(len(classes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/graphs.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    protocol = config["protocol"]
    # SSL validation edges are held out only for model selection. Downstream
    # evaluation embeds the complete observed target topology.
    target = load_graph(
        args.target, config["graphs"][args.target]["path"],
        seed=int(protocol["seed"]), use_full_edges=True,
    )
    labeled_index, y = labeled_nodes(target.data)
    model = GraphSAGE(
        int(protocol["input_dim"]), int(protocol["hidden_dim"]), int(protocol["output_dim"]),
        int(protocol["layers"]), float(protocol["dropout"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    embedding = embed(model.to(torch.device(f"cuda:{args.device}")), target, torch.device(f"cuda:{args.device}"))
    result = evaluate_embeddings(
        embedding[labeled_index.numpy()], y.numpy(), int(protocol["labels_per_class"]), int(protocol["seed"])
    )
    result.update({
        "target": args.target, "checkpoint": args.checkpoint, "checkpoint_step": int(checkpoint["step"]),
        "sources": checkpoint["metadata"]["sources"], "seed": int(checkpoint["metadata"]["seed"]),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

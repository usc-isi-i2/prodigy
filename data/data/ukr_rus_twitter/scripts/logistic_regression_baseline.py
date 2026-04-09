#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from data.midterm import (
    _apply_feature_subset,
    _apply_label_downsample,
    _build_stratified_node_splits,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a logistic-regression baseline on a labeled ukr_rus_twitter graph."
    )
    p.add_argument(
        "--graph",
        default="/scratch1/eibl/data/ukr_rus_twitter/graphs/retweet_graph_150files_minilm_hf03_political_labels.pt",
    )
    p.add_argument("--feature_subset", default="all")
    p.add_argument("--label_downsample", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument(
        "--no_standardize",
        action="store_true",
        help="Disable train-split standardization before logistic regression.",
    )
    p.add_argument("--out", default="")
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args()


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def print_metrics(split, metrics, n):
    parts = [f"{split} n={n}"]
    for k in ("accuracy", "f1", "roc_auc"):
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.4f}")
    print("  " + " ".join(parts))


def main():
    args = parse_args()
    raw = torch.load(args.graph, map_location="cpu")

    graph = Data(x=raw["x"].clone(), y=raw["y"].clone(), num_nodes=raw["x"].shape[0])
    graph.feature_names = list(raw.get("feature_names", []))
    graph.label_names = list(raw.get("label_names", []))

    if not graph.label_names:
        raise ValueError("Graph has no labels. Use a *_political_labels.pt graph.")

    graph.y = _apply_label_downsample(
        graph.y,
        graph.label_names,
        args.label_downsample,
        seed=int(args.seed),
    )
    graph = _apply_feature_subset(graph, args.feature_subset)

    x = graph.x.detach().cpu().numpy().astype(np.float64, copy=False)
    y = graph.y.detach().cpu().numpy().astype(np.int64, copy=False)

    labeled_mask = y >= 0
    if labeled_mask.sum() == 0:
        raise ValueError("No labeled nodes remain after downsampling.")

    splits = _build_stratified_node_splits(y, seed=int(args.seed))
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"Invalid split sizes after downsampling: train={train_idx.size}, val={val_idx.size}, test={test_idx.size}"
        )

    X_train = x[train_idx]
    X_val = x[val_idx]
    X_test = x[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    scaler = None
    if not args.no_standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        random_state=int(args.seed),
        max_iter=int(args.max_iter),
        C=float(args.C),
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    train_prob = clf.predict_proba(X_train)[:, 1]
    val_prob = clf.predict_proba(X_val)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    train_pred = (train_prob >= 0.5).astype(np.int64)
    val_pred = (val_prob >= 0.5).astype(np.int64)
    test_pred = (test_prob >= 0.5).astype(np.int64)

    train_metrics = compute_metrics(y_train, train_pred, train_prob)
    val_metrics = compute_metrics(y_val, val_pred, val_prob)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)

    print(f"Graph: {args.graph}")
    print(f"Feature subset: {args.feature_subset}")
    print(f"Feature dim: {X_train.shape[1]}")
    print(f"Labels: {graph.label_names}")
    print(f"Downsample: {args.label_downsample or '<none>'}")
    print("Split sizes:")
    print(f"  train={train_idx.size} val={val_idx.size} test={test_idx.size}")
    print("Metrics:")
    print_metrics("train", train_metrics, train_idx.size)
    print_metrics("val", val_metrics, val_idx.size)
    print_metrics("test", test_metrics, test_idx.size)

    feature_names = list(getattr(graph, "feature_names", []))
    if clf.coef_.shape[0] == 1 and feature_names and len(feature_names) == clf.coef_.shape[1]:
        coef = clf.coef_[0]
        order = np.argsort(np.abs(coef))[::-1][: min(args.topk, coef.shape[0])]
        print(f"Top {len(order)} features by |coef|:")
        for i in order:
            direction = graph.label_names[1] if coef[i] > 0 else graph.label_names[0]
            print(f"  {feature_names[i]}: coef={coef[i]:.6f} winner={direction}")

    if args.out:
        out = {
            "graph": args.graph,
            "feature_subset": args.feature_subset,
            "feature_dim": int(X_train.shape[1]),
            "label_names": list(graph.label_names),
            "label_downsample": args.label_downsample,
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "C": float(args.C),
            "standardized": not args.no_standardize,
            "split_sizes": {
                "train": int(train_idx.size),
                "val": int(val_idx.size),
                "test": int(test_idx.size),
            },
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
        }
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"Saved metrics: {args.out}")


if __name__ == "__main__":
    main()

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
    p.add_argument(
        "--audit_single_features",
        action="store_true",
        help="Fit one logistic-regression model per input feature on the same split.",
    )
    p.add_argument(
        "--audit_prefixes",
        default="",
        help="Comma-separated embedding prefix sizes to audit, e.g. '1,2,5,10,20'.",
    )
    p.add_argument(
        "--audit_topk",
        type=int,
        default=25,
        help="How many single-feature results to print.",
    )
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


def fit_logreg_eval(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    *,
    seed: int,
    max_iter: int,
    C: float,
    standardize: bool,
):
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        random_state=int(seed),
        max_iter=int(max_iter),
        C=float(C),
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    train_prob = clf.predict_proba(X_train)[:, 1]
    val_prob = clf.predict_proba(X_val)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    train_pred = (train_prob >= 0.5).astype(np.int64)
    val_pred = (val_prob >= 0.5).astype(np.int64)
    test_pred = (test_prob >= 0.5).astype(np.int64)

    return {
        "clf": clf,
        "metrics": {
            "train": compute_metrics(y_train, train_pred, train_prob),
            "val": compute_metrics(y_val, val_pred, val_prob),
            "test": compute_metrics(y_test, test_pred, test_prob),
        },
    }


def _metric_sort_key(row):
    val_auc = row["metrics"]["val"].get("roc_auc", float("-inf"))
    val_acc = row["metrics"]["val"].get("accuracy", float("-inf"))
    test_auc = row["metrics"]["test"].get("roc_auc", float("-inf"))
    test_acc = row["metrics"]["test"].get("accuracy", float("-inf"))
    return (val_auc, val_acc, test_auc, test_acc)


def run_single_feature_audit(
    x,
    feature_names,
    train_idx,
    val_idx,
    test_idx,
    y,
    *,
    seed: int,
    max_iter: int,
    C: float,
    standardize: bool,
    topk: int,
):
    rows = []
    for j in range(x.shape[1]):
        name = feature_names[j] if j < len(feature_names) else f"f{j}"
        result = fit_logreg_eval(
            x[train_idx, j:j + 1],
            x[val_idx, j:j + 1],
            x[test_idx, j:j + 1],
            y[train_idx],
            y[val_idx],
            y[test_idx],
            seed=seed,
            max_iter=max_iter,
            C=C,
            standardize=standardize,
        )
        coef = float(result["clf"].coef_[0, 0])
        rows.append(
            {
                "feature_index": int(j),
                "feature_name": name,
                "coef": coef,
                "metrics": result["metrics"],
            }
        )

    rows.sort(key=_metric_sort_key, reverse=True)
    print(f"Top {min(topk, len(rows))} single-feature models by val ROC-AUC / val accuracy:")
    for row in rows[: min(topk, len(rows))]:
        val_m = row["metrics"]["val"]
        test_m = row["metrics"]["test"]
        print(
            "  "
            f"{row['feature_name']}: "
            f"val_acc={val_m.get('accuracy', float('nan')):.4f} "
            f"val_auc={val_m.get('roc_auc', float('nan')):.4f} "
            f"test_acc={test_m.get('accuracy', float('nan')):.4f} "
            f"test_auc={test_m.get('roc_auc', float('nan')):.4f} "
            f"coef={row['coef']:.6f}"
        )
    return rows


def run_prefix_audit(
    x,
    feature_names,
    train_idx,
    val_idx,
    test_idx,
    y,
    prefix_sizes,
    *,
    seed: int,
    max_iter: int,
    C: float,
    standardize: bool,
):
    emb_idx = [i for i, name in enumerate(feature_names) if str(name).startswith("emb_")]
    if not emb_idx:
        raise ValueError("Prefix audit requested but no embedding features were found.")

    rows = []
    for n in prefix_sizes:
        if n <= 0:
            continue
        idx = emb_idx[: min(n, len(emb_idx))]
        result = fit_logreg_eval(
            x[train_idx][:, idx],
            x[val_idx][:, idx],
            x[test_idx][:, idx],
            y[train_idx],
            y[val_idx],
            y[test_idx],
            seed=seed,
            max_iter=max_iter,
            C=C,
            standardize=standardize,
        )
        rows.append(
            {
                "prefix_size": int(len(idx)),
                "metrics": result["metrics"],
            }
        )

    rows.sort(key=lambda row: row["prefix_size"])
    print("Embedding-prefix audit:")
    for row in rows:
        val_m = row["metrics"]["val"]
        test_m = row["metrics"]["test"]
        print(
            "  "
            f"first_{row['prefix_size']}_emb_dims: "
            f"val_acc={val_m.get('accuracy', float('nan')):.4f} "
            f"val_auc={val_m.get('roc_auc', float('nan')):.4f} "
            f"test_acc={test_m.get('accuracy', float('nan')):.4f} "
            f"test_auc={test_m.get('roc_auc', float('nan')):.4f}"
        )
    return rows


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
    result = fit_logreg_eval(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        seed=int(args.seed),
        max_iter=int(args.max_iter),
        C=float(args.C),
        standardize=not args.no_standardize,
    )
    clf = result["clf"]
    train_metrics = result["metrics"]["train"]
    val_metrics = result["metrics"]["val"]
    test_metrics = result["metrics"]["test"]

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

    single_feature_results = None
    if args.audit_single_features:
        single_feature_results = run_single_feature_audit(
            x=x,
            feature_names=feature_names,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            y=y,
            seed=int(args.seed),
            max_iter=int(args.max_iter),
            C=float(args.C),
            standardize=not args.no_standardize,
            topk=int(args.audit_topk),
        )

    prefix_results = None
    if args.audit_prefixes.strip():
        prefix_sizes = []
        for chunk in args.audit_prefixes.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            prefix_sizes.append(int(chunk))
        prefix_results = run_prefix_audit(
            x=x,
            feature_names=feature_names,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            y=y,
            prefix_sizes=prefix_sizes,
            seed=int(args.seed),
            max_iter=int(args.max_iter),
            C=float(args.C),
            standardize=not args.no_standardize,
        )

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
        if single_feature_results is not None:
            out["single_feature_audit"] = single_feature_results
        if prefix_results is not None:
            out["embedding_prefix_audit"] = prefix_results
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"Saved metrics: {args.out}")


if __name__ == "__main__":
    main()

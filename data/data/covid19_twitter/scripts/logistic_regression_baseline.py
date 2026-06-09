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
        description="Run a logistic-regression baseline on a labeled covid19_twitter graph."
    )
    p.add_argument(
        "--graph",
        default="/scratch1/eibl/data/covid19_twitter/graphs/retweet_graph_parquet.pt",
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
                "prefix_dim": int(len(idx)),
                "metrics": result["metrics"],
            }
        )

    rows.sort(key=_metric_sort_key, reverse=True)
    print("Embedding prefix audit:")
    for row in rows:
        val_m = row["metrics"]["val"]
        test_m = row["metrics"]["test"]
        print(
            "  "
            f"emb[:{row['prefix_dim']}]: "
            f"val_acc={val_m.get('accuracy', float('nan')):.4f} "
            f"val_auc={val_m.get('roc_auc', float('nan')):.4f} "
            f"test_acc={test_m.get('accuracy', float('nan')):.4f} "
            f"test_auc={test_m.get('roc_auc', float('nan')):.4f}"
        )
    return rows


def main():
    args = parse_args()

    raw = torch.load(args.graph, map_location="cpu")
    if isinstance(raw, Data):
        graph = {"data": raw}
        graph["x"] = raw.x
        graph["y"] = raw.y
        graph["feature_names"] = list(getattr(raw, "feature_names", []))
    else:
        graph = raw

    x = graph["x"]
    y = graph["y"]
    feature_names = list(graph.get("feature_names", []))

    x, feature_names = _apply_feature_subset(x, feature_names, args.feature_subset)
    y, keep_mask, downsample_meta = _apply_label_downsample(y, args.label_downsample)
    x = x[keep_mask]

    train_idx, val_idx, test_idx = _build_stratified_node_splits(
        y,
        seed=int(args.seed),
    )

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    train_np = train_idx.detach().cpu().numpy()
    val_np = val_idx.detach().cpu().numpy()
    test_np = test_idx.detach().cpu().numpy()

    print(f"Graph: {args.graph}")
    print(f"Feature subset: {args.feature_subset}")
    print(f"Feature dim: {x_np.shape[1]}")
    print(f"Labeled nodes: {len(y_np)}")
    print(f"Split sizes: train={len(train_np)} val={len(val_np)} test={len(test_np)}")
    if downsample_meta["applied"]:
        print(f"Label downsample: {downsample_meta}")

    result = fit_logreg_eval(
        x_np[train_np],
        x_np[val_np],
        x_np[test_np],
        y_np[train_np],
        y_np[val_np],
        y_np[test_np],
        seed=int(args.seed),
        max_iter=int(args.max_iter),
        C=float(args.C),
        standardize=not args.no_standardize,
    )

    print("Metrics")
    print_metrics("train", result["metrics"]["train"], len(train_np))
    print_metrics("val", result["metrics"]["val"], len(val_np))
    print_metrics("test", result["metrics"]["test"], len(test_np))

    audit_rows = None
    if args.audit_single_features:
        audit_rows = run_single_feature_audit(
            x_np,
            feature_names,
            train_np,
            val_np,
            test_np,
            y_np,
            seed=int(args.seed),
            max_iter=int(args.max_iter),
            C=float(args.C),
            standardize=not args.no_standardize,
            topk=int(args.audit_topk),
        )

    prefix_rows = None
    if args.audit_prefixes.strip():
        prefix_sizes = [int(part.strip()) for part in args.audit_prefixes.split(",") if part.strip()]
        prefix_rows = run_prefix_audit(
            x_np,
            feature_names,
            train_np,
            val_np,
            test_np,
            y_np,
            prefix_sizes,
            seed=int(args.seed),
            max_iter=int(args.max_iter),
            C=float(args.C),
            standardize=not args.no_standardize,
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {
            "graph": args.graph,
            "feature_subset": args.feature_subset,
            "label_downsample": args.label_downsample,
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "C": float(args.C),
            "standardize": not args.no_standardize,
            "metrics": result["metrics"],
            "split_sizes": {
                "train": int(len(train_np)),
                "val": int(len(val_np)),
                "test": int(len(test_np)),
            },
            "feature_dim": int(x_np.shape[1]),
            "downsample_meta": downsample_meta,
            "audit_single_features": audit_rows,
            "audit_prefixes": prefix_rows,
            "topk_feature_names": feature_names[: args.topk],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()

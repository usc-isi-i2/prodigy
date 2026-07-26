#!/usr/bin/env python3
"""Feature-quality probe: do raw node features linearly predict the label?

No graph structure is used — just logistic regression from node features to the
node label. This distinguishes the two explanations for the feature-ablation
finding (that NM is permute-invariant / zero-fragile):

  * test AUC >> 0.5  -> features carry real label signal; a model *could* use
    them, so NM ignoring feature content is an inductive-bias story.
  * test AUC ~ 0.5   -> the features themselves are uninformative/broken, which
    would be the actual problem.

Self-contained (torch + sklearn); does not depend on the per-dataset helpers.
Handles unlabeled nodes marked with y < 0 (dropped) and class imbalance
(class_weight="balanced", macro metrics, majority baseline reported).

Usage:
  python3 feature_label_probe.py --graph <retweet_graph.pt> [--name twibot20] [--out probe.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_xy(path: str):
    g = torch.load(path, map_location="cpu")
    if isinstance(g, dict):
        if "data" in g and hasattr(g["data"], "x"):
            x, y = g["data"].x, g["data"].y
        else:
            x, y = g["x"], g["y"]
    else:  # torch_geometric Data
        x, y = g.x, g.y
    return x.float().numpy(), y.long().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    name = args.name or args.graph
    X, y = load_xy(args.graph)
    labeled = y >= 0
    X, y = X[labeled], y[labeled]
    classes, counts = np.unique(y, return_counts=True)
    majority = counts.max() / counts.sum()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    clf = LogisticRegression(
        max_iter=args.max_iter, C=args.C, class_weight="balanced", solver="lbfgs"
    ).fit(Xtr, ytr)

    prob = clf.predict_proba(Xte)
    pred = clf.predict(Xte)
    if len(classes) == 2:
        auc = roc_auc_score(yte, prob[:, 1])
    else:
        y1hot = np.eye(len(classes))[np.searchsorted(classes, yte)]
        auc = roc_auc_score(y1hot, prob, multi_class="ovr", average="macro")
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    result = {
        "name": name,
        "graph": args.graph,
        "feature_dim": int(X.shape[1]),
        "n_labeled": int(len(y)),
        "n_classes": int(len(classes)),
        "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
        "majority_acc": float(majority),
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
        "test_roc_auc": float(auc),
    }
    print(
        f"{name:<20} dim={result['feature_dim']} n={result['n_labeled']:>7} "
        f"classes={len(classes)}  AUC={auc:.4f}  acc={acc:.4f} "
        f"(majority {majority:.4f})  f1_macro={f1:.4f}"
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

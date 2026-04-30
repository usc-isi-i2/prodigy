"""
Logistic regression analysis of stats features vs. political leaning in the midterm graph.

Usage:
    python scripts/analysis/midterm_feature_regression.py \
        --graph /scratch1/eibl/data/midterm/graphs/graph_data.pt

The script reports per-feature AUC and accuracy (univariate) plus a joint model
over all stats features (everything that is not an emb_* dimension).
"""

import argparse
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler


def load_graph(path: str):
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        x = raw["x"] if isinstance(raw["x"], torch.Tensor) else raw["data"].x
        y = raw["y"] if "y" in raw else raw["data"].y
        feature_names = raw.get("feature_names", [])
        if not feature_names and hasattr(raw.get("data"), "feature_names"):
            feature_names = raw["data"].feature_names
    else:
        # torch_geometric Data object
        x = raw.x
        y = raw.y
        feature_names = getattr(raw, "feature_names", [])
    return x, y, list(feature_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Path to the .pt graph file")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading graph from {args.graph} ...")
    x, y, feature_names = load_graph(args.graph)

    x = x.numpy() if isinstance(x, torch.Tensor) else np.array(x)
    y = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)

    n_total = len(y)
    labeled_mask = (y >= 0) & (y < 2)
    x_lab = x[labeled_mask]
    y_lab = y[labeled_mask].astype(int)
    print(f"Nodes: {n_total} total, {labeled_mask.sum()} labeled  "
          f"(rep={( y_lab==0).sum()}, dem={(y_lab==1).sum()})")

    if not feature_names or len(feature_names) != x.shape[1]:
        feature_names = [f"f{i}" for i in range(x.shape[1])]

    stats_idx = [i for i, n in enumerate(feature_names) if not n.startswith("emb_")]
    stats_names = [feature_names[i] for i in stats_idx]
    x_stats = x_lab[:, stats_idx]

    print(f"\nStats features ({len(stats_names)}): {stats_names}\n")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_stats)

    def fit_logreg(X):
        clf = LogisticRegression(max_iter=args.max_iter, random_state=args.seed, solver="lbfgs")
        clf.fit(X, y_lab)
        proba = clf.predict_proba(X)[:, 1]
        pred = clf.predict(X)
        auc = roc_auc_score(y_lab, proba)
        acc = accuracy_score(y_lab, pred)
        return auc, acc

    # --- univariate ---
    print(f"{'Feature':<22}  {'AUC':>6}  {'Acc':>6}")
    print("-" * 40)
    results = []
    for j, name in enumerate(stats_names):
        auc, acc = fit_logreg(x_scaled[:, j : j + 1])
        results.append((auc, acc, name))

    for auc, acc, name in sorted(results, reverse=True):
        print(f"{name:<22}  {auc:.4f}  {acc:.4f}")

    # --- joint ---
    print("-" * 40)
    auc, acc = fit_logreg(x_scaled)
    print(f"{'[all stats joint]':<22}  {auc:.4f}  {acc:.4f}")

    # --- coefficients of joint model ---
    clf_joint = LogisticRegression(max_iter=args.max_iter, random_state=args.seed, solver="lbfgs")
    clf_joint.fit(x_scaled, y_lab)
    coefs = clf_joint.coef_[0]
    print(f"\nJoint model coefficients (positive = dem, negative = rep):")
    print(f"{'Feature':<22}  {'Coef':>8}")
    print("-" * 33)
    for name, coef in sorted(zip(stats_names, coefs), key=lambda t: -abs(t[1])):
        print(f"{name:<22}  {coef:+.4f}")


if __name__ == "__main__":
    main()

"""
Logistic regression analysis of node features vs. political leaning in the midterm graph.

Usage:
    python scripts/analysis/midterm_feature_regression.py \
        --graph /scratch1/eibl/data/midterm/graphs/graph_data.pt \
        --features stats_only        # default
        --features emb_only
        --features all

The script reports per-feature AUC (univariate) plus a joint model over the selected features.
"""

import argparse

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
        x = raw.x
        y = raw.y
        feature_names = getattr(raw, "feature_names", [])
    return x, y, list(feature_names)


def select_features(x, feature_names, mode):
    if mode == "stats_only":
        idx = [i for i, n in enumerate(feature_names) if not n.startswith("emb_")]
    elif mode == "emb_only":
        idx = [i for i, n in enumerate(feature_names) if n.startswith("emb_")]
    else:  # all
        idx = list(range(len(feature_names)))
    return x[:, idx], [feature_names[i] for i in idx]


def fit_logreg(X, y, max_iter, seed):
    clf = LogisticRegression(max_iter=max_iter, random_state=seed, solver="lbfgs")
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    pred = clf.predict(X)
    return clf, roc_auc_score(y, proba), accuracy_score(y, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Path to the .pt graph file")
    parser.add_argument(
        "--features",
        choices=["stats_only", "emb_only", "all"],
        default="stats_only",
        help="Which feature subset to analyse",
    )
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading graph from {args.graph} ...")
    x, y, feature_names = load_graph(args.graph)

    x = x.numpy() if isinstance(x, torch.Tensor) else np.array(x)
    y = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)

    if not feature_names or len(feature_names) != x.shape[1]:
        feature_names = [f"f{i}" for i in range(x.shape[1])]

    labeled_mask = (y >= 0) & (y < 2)
    x_lab = x[labeled_mask]
    y_lab = y[labeled_mask].astype(int)
    print(
        f"Nodes: {len(y)} total, {labeled_mask.sum()} labeled  "
        f"(rep={( y_lab==0).sum()}, dem={(y_lab==1).sum()})"
    )

    x_sel, names = select_features(x_lab, feature_names, args.features)
    print(f"\nFeature mode: {args.features}  ({len(names)} dims)\n")

    if len(names) == 0:
        print(f"No features for mode '{args.features}' — skipping.")
        return

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_sel)

    # --- univariate (skip for emb_only / all since there are hundreds of dims) ---
    if args.features == "stats_only":
        print(f"{'Feature':<22}  {'AUC':>6}  {'Acc':>6}")
        print("-" * 40)
        results = []
        for j, name in enumerate(names):
            _, auc, acc = fit_logreg(x_scaled[:, j : j + 1], y_lab, args.max_iter, args.seed)
            results.append((auc, acc, name))
        for auc, acc, name in sorted(results, reverse=True):
            print(f"{name:<22}  {auc:.4f}  {acc:.4f}")
        print("-" * 40)

    # --- joint ---
    clf_joint, auc, acc = fit_logreg(x_scaled, y_lab, args.max_iter, args.seed)
    label = f"[{args.features} joint]"
    print(f"{label:<22}  {auc:.4f}  {acc:.4f}")

    # --- joint coefficients (stats_only only, emb has too many dims) ---
    if args.features == "stats_only":
        coefs = clf_joint.coef_[0]
        print(f"\nJoint model coefficients (positive = dem, negative = rep):")
        print(f"{'Feature':<22}  {'Coef':>8}")
        print("-" * 33)
        for name, coef in sorted(zip(names, coefs), key=lambda t: -abs(t[1])):
            print(f"{name:<22}  {coef:+.4f}")


if __name__ == "__main__":
    main()

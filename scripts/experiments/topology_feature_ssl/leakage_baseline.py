#!/usr/bin/env python3
"""Leakage-control baseline (topology_feature_ssl README "Leakage control").

E1/E2 feed raw structural features (degree/PageRank/...) as input, and the
regression targets `followers` (≈ in-degree) and `statuses` (≈ out-degree) are
near-copies of those inputs. An encoder can therefore "win" structure-linked
regression trivially by PASSTHROUGH, not by learning. This script measures that
passthrough ceiling directly: linear-probe the raw
[in_deg, out_deg, log_deg, k_core, pagerank, clustering] vector onto each
regression target, NO encoder. E1/E2 count as "learned structure" only if the
frozen rep BEATS this baseline (README reading: "E1 - B0, against the baseline").

Output: scripts/plotting/topology_feature_ssl/data/leakage_baseline.csv
        (dataset, target, spearman, n) — the Δ reference for T1's structure-linked column.

Run on Tucker (prodigy env; needs torch, scikit-learn, scipy, networkx).
    python scripts/experiments/topology_feature_ssl/leakage_baseline.py \
        --data-root /dataMeR1/phil/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from data.structural_features import STRUCTURAL_FEATURE_NAMES, compute_structural_features

# dataset -> (root_name, graph_filename); the focused datasets that carry node_targets.
DATASETS = {
    "midterm": ("midterm", "retweet_graph_parquet.pt"),
    "ukr_rus_twitter": ("ukr_rus_twitter", "retweet_graph_parquet.pt"),
    "covid19_twitter": ("covid19_twitter", "retweet_graph_parquet.pt"),
    "twibot20": ("twibot20", "retweet_graph.pt"),
    "election2020": ("election2020", "retweet_graph.pt"),
}
DEFAULT_TARGETS = ["followers_count", "statuses_count", "account_age_days"]


def _load_raw(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _node_targets(raw: dict) -> dict[str, torch.Tensor]:
    nt = raw.get("node_targets")
    return nt if isinstance(nt, dict) else {}


def _cv_spearman(x: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5) -> float:
    """Out-of-fold Spearman of a Ridge fit (structural feats -> target)."""
    preds = np.zeros_like(y, dtype=float)
    kf = KFold(n_splits=min(folds, max(2, len(y) // 50)), shuffle=True, random_state=seed)
    for tr, te in kf.split(x):
        scaler = StandardScaler().fit(x[tr])
        model = Ridge(alpha=1.0).fit(scaler.transform(x[tr]), y[tr])
        preds[te] = model.predict(scaler.transform(x[te]))
    rho = spearmanr(preds, y).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/dataMeR1/phil/data")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--transform", default="log1p", choices=["none", "log1p"])
    ap.add_argument("--max-nx-nodes", type=int, default=400000,
                    help="Skip networkx features (k_core/pagerank/clustering) above this node count.")
    ap.add_argument("--out", default="scripts/plotting/topology_feature_ssl/data/leakage_baseline.csv")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    rows = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if name not in DATASETS:
            print(f"[leakage] skip unknown dataset {name}")
            continue
        root_name, fname = DATASETS[name]
        path = Path(args.data_root) / root_name / "graphs" / fname
        if not path.exists():
            print(f"[leakage] skip {name}: missing {path}")
            continue
        raw = _load_raw(path)
        x = raw["x"]
        num_nodes = int(x.shape[0])
        edge_index = raw.get("edge_index")
        if edge_index is None:
            print(f"[leakage] skip {name}: no edge_index")
            continue
        print(f"[leakage] {name}: {num_nodes} nodes — computing structural features")
        feats = compute_structural_features(
            edge_index, num_nodes, standardize=False, max_nx_nodes=args.max_nx_nodes
        ).numpy()

        node_targets = _node_targets(raw)
        for target in targets:
            tvec = node_targets.get(target)
            if tvec is None:
                continue
            y = tvec.float().numpy()
            mask = np.isfinite(y)
            if mask.sum() < 100:
                print(f"[leakage] {name}/{target}: only {int(mask.sum())} labeled — skip")
                continue
            yy = y[mask]
            if args.transform == "log1p":
                yy = np.log1p(np.clip(yy, 0, None))
            rho = _cv_spearman(feats[mask], yy)
            rows.append({"dataset": name, "target": target, "spearman": rho,
                         "n": int(mask.sum())})
            print(f"[leakage] {name}/{target}: raw-structural Spearman={rho:.3f} "
                  f"(n={int(mask.sum())})")

    if not rows:
        raise SystemExit("[leakage] no (dataset,target) pairs produced — check paths/targets.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "target", "spearman", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[leakage] wrote {out} ({len(rows)} rows). "
          f"Structural feature order: {STRUCTURAL_FEATURE_NAMES}")
    print("[leakage] E1/E2 'learned structure' ⇒ frozen-rep Spearman > these on "
          "structure-linked targets (followers/statuses).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

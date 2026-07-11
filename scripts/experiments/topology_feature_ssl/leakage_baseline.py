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
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# repo root on sys.path so `data.*` imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.structural_features import compute_structural_features, structural_feature_names

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
    """Out-of-fold Spearman of a full-data Ridge fit (structural feats -> target).
    Reference ceiling only — NOT comparable to the few-shot frozen-rep eval."""
    preds = np.zeros_like(y, dtype=float)
    kf = KFold(n_splits=min(folds, max(2, len(y) // 50)), shuffle=True, random_state=seed)
    for tr, te in kf.split(x):
        scaler = StandardScaler().fit(x[tr])
        model = Ridge(alpha=1.0).fit(scaler.transform(x[tr]), y[tr])
        preds[te] = model.predict(scaler.transform(x[te]))
    rho = spearmanr(preds, y).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _fewshot_spearman(x: np.ndarray, y: np.ndarray, shots: int, n_query: int,
                      episodes: int, seed: int = 0) -> float:
    """Shot-MATCHED leakage ceiling: episodic linear regression on the raw
    structural features, using the SAME support size as the frozen-rep benchmark
    (n_shots). Each episode fits Ridge on `shots` random support nodes and predicts
    a `n_query` query set; predictions are accumulated across episodes and scored
    once (mirroring the benchmark's accumulated Spearman). This is the fair
    passthrough baseline the frozen rep must beat to claim it learned structure."""
    n = len(y)
    if n < shots + n_query:
        return float("nan")
    rng = np.random.default_rng(seed)
    preds: list[float] = []
    trues: list[float] = []
    for _ in range(episodes):
        pick = rng.choice(n, size=shots + n_query, replace=False)
        sup, qry = pick[:shots], pick[shots:]
        scaler = StandardScaler().fit(x[sup])
        model = Ridge(alpha=1.0).fit(scaler.transform(x[sup]), y[sup])
        preds.extend(model.predict(scaler.transform(x[qry])).tolist())
        trues.extend(y[qry].tolist())
    rho = spearmanr(preds, trues).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/dataMeR1/phil/data")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--transform", default="log1p", choices=["none", "log1p"])
    ap.add_argument("--mode", default="directed3", choices=["directed3", "directed6"],
                    help="Structural feature set — must MATCH the E1/E2 encoders' "
                         "--structural_features to be their fair passthrough ceiling.")
    ap.add_argument("--features", default="structural", choices=["structural", "raw"],
                    help="Ridge inputs: 'structural' = compute_structural_features "
                         "(the E1/E2 passthrough ceiling); 'raw' = the raw node embeddings "
                         "raw['x'] the encoders actually consume — i.e. the features_only "
                         "floor for the frozen-probe matrix (pretrain_probe_matrix).")
    ap.add_argument("--shots", type=int, default=10,
                    help="Support size for the shot-matched ceiling (match the benchmark's n_shots).")
    ap.add_argument("--n-query", type=int, default=12, help="Query size per episode (match the benchmark).")
    ap.add_argument("--episodes", type=int, default=500, help="Number of few-shot episodes to accumulate.")
    ap.add_argument("--skip-fulldata", action="store_true",
                    help="Skip the full-data Ridge reference (_cv_spearman). It fits on ALL "
                         "labeled nodes (O(n·d²)) — minutes-to-hours on the 23M-node graphs — "
                         "and is NOT comparable to the few-shot floor anyway. Use for large "
                         "graphs / the features_only floor.")
    ap.add_argument("--out", default="",
                    help="Output CSV; defaults to a features-mode-specific path.")
    args = ap.parse_args()

    if not args.out:
        args.out = (
            "scripts/plotting/node_regression/data/features_only_floor.csv"
            if args.features == "raw"
            else "scripts/plotting/topology_feature_ssl/data/leakage_baseline.csv"
        )

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
        raw_mode = args.features == "raw"
        if raw_mode:
            print(f"[leakage] {name}: {num_nodes} nodes — using raw node features raw['x'] "
                  f"(dim {int(x.shape[1])}) as the features_only floor "
                  f"(indexed per-target; the full float32 matrix is never materialized)")
            feats = None
        else:
            print(f"[leakage] {name}: {num_nodes} nodes — computing {args.mode} structural features")
            feats = compute_structural_features(
                edge_index, num_nodes, mode=args.mode, standardize=False
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
            if raw_mode:
                # Gather only the labeled rows out of the (mmap'd) feature tensor. The
                # covid graph has ~23M nodes, so x.float().numpy() would materialize a
                # ~70GB matrix; indexing first keeps peak RAM at (n_labeled x 768).
                idx = np.nonzero(mask)[0]
                xm = x[torch.from_numpy(idx)].float().numpy()
            else:
                xm = feats[mask]
            rho_fs = _fewshot_spearman(xm, yy, args.shots, args.n_query, args.episodes)
            rho_full = float("nan") if args.skip_fulldata else _cv_spearman(xm, yy)
            rows.append({"dataset": name, "target": target,
                         "spearman": rho_fs,            # shot-matched (primary, fair)
                         "spearman_fulldata": rho_full,  # full-data reference (skippable)
                         "shots": args.shots, "n": int(mask.sum())})
            ref = "skipped" if args.skip_fulldata else f"{rho_full:.3f}"
            print(f"[leakage] {name}/{target}: {args.shots}-shot Spearman={rho_fs:.3f} "
                  f"(full-data ref={ref}, n={int(mask.sum())})")

    if not rows:
        raise SystemExit("[leakage] no (dataset,target) pairs produced — check paths/targets.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "target", "spearman",
                                           "spearman_fulldata", "shots", "n"])
        w.writeheader()
        w.writerows(rows)
    if args.features == "raw":
        feat_desc = "raw node features raw['x'] (features_only floor)"
    else:
        feat_desc = f"structural features ({args.mode}): {structural_feature_names(args.mode)}"
    print(f"\n[leakage] wrote {out} ({len(rows)} rows). Inputs: {feat_desc}")
    print(f"[leakage] 'spearman' is the {args.shots}-shot ceiling, fair vs the frozen-rep "
          f"{args.shots}-shot eval.")
    if args.features == "raw":
        print("[leakage] features_only floor: a pretrained row 'learned' beyond raw features "
              "⇒ its frozen-rep Spearman > this on the same (dataset,target).")
    else:
        print("[leakage] E1/E2 'learned structure' ⇒ frozen-rep > this on structure-linked "
              "targets (followers/statuses).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

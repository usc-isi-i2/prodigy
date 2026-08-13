#!/usr/bin/env python3
"""Trivial 'no-pretraining' baselines — the absolute anchor the arm-vs-arm
comparison was missing. For each dataset, a 10-shot linear probe on the RAW 768-d
bio embedding (no encoder, no pretraining), shot-matched to the benchmark:

  * regression  -> episodic Ridge (bio emb -> log1p target), accumulated Spearman
  * classification -> episodic logistic (bio emb -> label), accumulated ROC-AUC

If no pretrained arm beats `raw_feat`, then on these graphs SSL pretraining does
not buy transfer over just using the features — a real (if negative) result.

Output: scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/data/trivial_baselines.csv
        (model=raw_feat, dataset, task, target, metric, n)

Run on Tucker (needs torch, scikit-learn, scipy).
    python scripts/experiments/setup/topology_feature_ssl/trivial_baselines.py --data-root /dataMeR1/phil/data
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATASETS = {
    "midterm": ("midterm", "retweet_graph_parquet.pt"),
    "ukr_rus_twitter": ("ukr_rus_twitter", "retweet_graph_parquet.pt"),
    "covid19_twitter": ("covid19_twitter", "retweet_graph_parquet.pt"),
    "twibot20": ("twibot20", "retweet_graph.pt"),
    "election2020": ("election2020", "retweet_graph.pt"),
}
REG6 = ["followers_count", "friends_count", "statuses_count",
        "favourites_count", "listed_count", "account_age_days"]


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _reg_fewshot(x, y, shots, n_query, episodes, seed=0):
    n = len(y)
    if n < shots + n_query:
        return float("nan")
    rng = np.random.default_rng(seed)
    preds, trues = [], []
    for _ in range(episodes):
        pick = rng.choice(n, size=shots + n_query, replace=False)
        sup, qry = pick[:shots], pick[shots:]
        sc = StandardScaler().fit(x[sup])
        m = Ridge(alpha=1.0).fit(sc.transform(x[sup]), y[sup])
        preds.extend(m.predict(sc.transform(x[qry])).tolist()); trues.extend(y[qry].tolist())
    rho = spearmanr(preds, trues).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _cls_fewshot(x, y, shots, n_query, episodes, seed=0):
    """Balanced few-shot logistic: `shots` support + `n_query` query per class."""
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    per = {c: np.where(y == c)[0] for c in classes}
    if any(len(idx) < shots + n_query for idx in per.values()):
        return float("nan")
    rng = np.random.default_rng(seed)
    scores, weights = [], []
    for _ in range(episodes):
        sup, qry = [], []
        for c in classes:
            pick = rng.choice(per[c], size=shots + n_query, replace=False)
            sup.extend(pick[:shots]); qry.extend(pick[shots:])
        sup, qry = np.array(sup), np.array(qry)
        sc = StandardScaler().fit(x[sup])
        try:
            m = LogisticRegression(max_iter=200).fit(sc.transform(x[sup]), y[sup])
            p = m.predict_proba(sc.transform(x[qry]))[:, 1]
            scores.append(roc_auc_score(y[qry], p)); weights.append(len(qry))
        except Exception:
            continue
    return float(np.average(scores, weights=weights)) if scores else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/dataMeR1/phil/data")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--targets", default=",".join(REG6))
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--n-query", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default="scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/data/trivial_baselines.csv")
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    rows = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if name not in DATASETS:
            continue
        root, fname = DATASETS[name]
        path = Path(args.data_root) / root / "graphs" / fname
        if not path.exists():
            print(f"[raw_feat] skip {name}: missing {path}"); continue
        raw = _load(path)
        x = raw["x"].float().numpy()
        print(f"[raw_feat] {name}: {x.shape[0]} nodes, {x.shape[1]}-d features")
        nt = raw.get("node_targets") or {}
        for target in targets:
            tv = nt.get(target)
            if tv is None:
                continue
            y = tv.float().numpy(); mask = np.isfinite(y)
            if mask.sum() < args.shots + args.n_query:
                continue
            yy = np.log1p(np.clip(y[mask], 0, None))
            rho = _reg_fewshot(x[mask], yy, args.shots, args.n_query, args.episodes)
            rows.append({"model": "raw_feat", "dataset": name, "task": "regression",
                         "target": target, "metric": rho, "n": int(mask.sum())})
            print(f"[raw_feat] {name}/{target}: reg {args.shots}-shot Spearman={rho:.3f}")
        y = raw.get("y")
        if y is not None:
            y = y.numpy()
            lab = y >= 0
            if lab.sum() >= 2 * (args.shots + args.n_query):
                auc = _cls_fewshot(x[lab], y[lab].astype(int), args.shots, args.n_query, args.episodes)
                rows.append({"model": "raw_feat", "dataset": name, "task": "classification",
                             "target": "", "metric": auc, "n": int(lab.sum())})
                print(f"[raw_feat] {name}: cls {args.shots}-shot ROC-AUC={auc:.3f}")
    if not rows:
        raise SystemExit("[raw_feat] no rows produced")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "dataset", "task", "target", "metric", "n"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[raw_feat] wrote {out} ({len(rows)} rows) — the 'just use the features' anchor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

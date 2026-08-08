#!/usr/bin/env python3
"""Compare predictor sets under leave-one-graph-out transfer-AUC prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "scripts/experiments/analysis/graph_divergence/data/graph_divergence_data.json"
TRANSFER = ROOT / "scripts/experiments/analysis/nm_single_source_matrix_facebook/data/nm_single_source_matrix_9x9_long.csv"
DEFAULT_EXTENDED = Path(__file__).resolve().parent / "data/extended_predictors.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "data/extended"


SOURCE_FEATURES = [
    "n_nodes", "n_edges", "density", "degree_assortativity", "avg_clustering_approx",
    "in_degree_gini", "out_degree_gini", "in_degree_max", "out_degree_max",
    "reciprocity", "largest_wcc_frac", "largest_scc_frac", "isolated_fraction",
    "missing_bio_rate", "feature_norm_mean", "feature_effective_dim",
    "edge_bio_coverage", "feature_homophily", "feature_homophily_random", "dirichlet_energy",
]


def model(kind: str):
    if kind == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 4, 30)),
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3,
                            max_features=0.7, random_state=20260807, n_jobs=-1),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extended", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    base = json.loads(BASE.read_text())
    ext = json.loads(args.extended.read_text())
    names = base["graphs"]
    if ext["graphs"] != names:
        raise ValueError("graph order mismatch")
    transfer = pd.read_csv(TRANSFER)
    auc = (transfer[transfer.metric == "roc_auc"].pivot(index="train", columns="test", values="value")
           .reindex(index=names, columns=names).to_numpy(float))
    pairwise = {**base["pairwise"], **ext["pairwise"]}
    full_pairwise = {k: np.asarray(v, float) for k, v in pairwise.items()
                     if np.isfinite(np.asarray(v, float)).all()}
    original = list(base["pairwise"])
    extended = [k for k in full_pairwise if k not in original]
    handpicked = [k for k in ["proxy_a_distance", "neighbor_mean_projected_frechet",
                              "center_plus_neighbor_mean_proxy_a_distance",
                              "embedding_topic_js_distance", "indegree_ks"] if k in full_pairwise]

    rows, feature_rows = [], []
    for source in range(len(names)):
        for target in range(len(names)):
            if source == target:
                continue
            row = {"source": names[source], "target": names[target], "auc": auc[source, target]}
            features = {}
            for key in SOURCE_FEATURES:
                sv = base["per_graph"][names[source]].get(key)
                tv = base["per_graph"][names[target]].get(key)
                features[f"source__{key}"] = sv
                features[f"target__{key}"] = tv
                features[f"signed__{key}"] = None if sv is None or tv is None else sv - tv
                features[f"gap__{key}"] = None if sv is None or tv is None else abs(sv - tv)
            for key, matrix in full_pairwise.items():
                features[f"pair__{key}"] = matrix[source, target]
            rows.append(row); feature_rows.append(features)
    meta, features = pd.DataFrame(rows), pd.DataFrame(feature_rows, dtype=float)

    source_cols = [c for c in features if c.startswith("source__")]
    graph_cols = [c for c in features if not c.startswith("pair__")]
    sets = {
        "source_only": source_cols,
        "original_pairwise": [f"pair__{k}" for k in original],
        "handpicked": source_cols + [f"pair__{k}" for k in handpicked],
        "source_plus_original": source_cols + [f"pair__{k}" for k in original],
        "all_graph_descriptors": graph_cols,
        "all_full_coverage": graph_cols + [f"pair__{k}" for k in original + extended],
    }
    predictions = []
    for set_name, columns in sets.items():
        for algorithm in ("ridge", "extra_trees"):
            for held_graph in names:
                test = meta.target == held_graph
                train = (meta.target != held_graph) & (meta.source != held_graph)
                estimator = model(algorithm)
                estimator.fit(features.loc[train, columns], meta.loc[train, "auc"])
                pred = estimator.predict(features.loc[test, columns])
                for index, value in zip(meta.index[test], pred):
                    predictions.append({"feature_set": set_name, "algorithm": algorithm,
                                        "held_graph": held_graph, "source": meta.loc[index, "source"],
                                        "truth": meta.loc[index, "auc"], "prediction": float(value)})
    pred = pd.DataFrame(predictions)
    ranking = []
    for (feature_set, algorithm), group in pred.groupby(["feature_set", "algorithm"]):
        target_rhos, regrets, hits = [], [], 0
        for _, target in group.groupby("held_graph"):
            target_rhos.append(spearmanr(target.prediction, target.truth).statistic)
            chosen = target.iloc[int(np.argmax(target.prediction.to_numpy()))]
            best = target.truth.max()
            regrets.append(float(best - chosen.truth)); hits += bool(np.isclose(chosen.truth, best))
        ranking.append({"feature_set": feature_set, "algorithm": algorithm,
                        "n_features": len(sets[feature_set]),
                        "mean_target_spearman": float(np.mean(target_rhos)),
                        "mae": float(mean_absolute_error(group.truth, group.prediction)),
                        "rmse": float(mean_squared_error(group.truth, group.prediction) ** 0.5),
                        "r2": float(r2_score(group.truth, group.prediction)),
                        "top1_hits": hits, "mean_regret": float(np.mean(regrets))})
    ranking = pd.DataFrame(ranking)
    ranking["priority_score"] = ranking.mean_target_spearman - ranking.mae
    ranking = ranking.sort_values("priority_score", ascending=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out_dir / "model_ranking_logo.csv", index=False)
    pred.to_csv(args.out_dir / "model_predictions_logo.csv", index=False)


if __name__ == "__main__":
    main()

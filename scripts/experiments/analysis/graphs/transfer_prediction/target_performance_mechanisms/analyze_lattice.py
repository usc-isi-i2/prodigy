#!/usr/bin/env python3
"""Audit the completed singleton/pair/LOO lattice; descriptive, not causal fitting."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[2]
ALIASES = {"covid19_twitter": "covid", "ukr_rus_twitter": "ukr_rus", "cp_hk_twitter": "cp_hk"}


def key(name):
    return ALIASES.get(name, name)


def variance_parts(frame, outcome, model="model_id", target="target"):
    """Orthogonal descriptive ANOVA on a COMPLETE rectangular matrix only."""
    matrix = frame.pivot(index=model, columns=target, values=outcome)
    if matrix.isna().any().any():
        raise ValueError("Variance decomposition requires a complete rectangular matrix")
    y = matrix.to_numpy()
    grand = y.mean()
    target_effect = y.mean(axis=0, keepdims=True) - grand
    model_effect = y.mean(axis=1, keepdims=True) - grand
    residual = y - grand - target_effect - model_effect
    total = ((y - grand) ** 2).sum()
    return {
        "cells": int(y.size),
        "target_share": float((target_effect**2).sum() * y.shape[0] / total),
        "model_share": float((model_effect**2).sum() * y.shape[1] / total),
        "interaction_plus_noise_share": float((residual**2).sum() / total),
        "interaction_rmse": float(np.sqrt((residual**2).mean())),
        "note": "In-sample sums of squares; interaction includes training/evaluation noise. Not an out-of-target forecast or causal partition.",
    }


def validate_cls(frame):
    if frame[["model_id", "dataset"]].duplicated().any():
        raise ValueError("Duplicate classification model-target cells")
    expected_targets = {"covid_political", "election2020", "facebook_page_reference", "twibot20", "ukr_rus_suspended"}
    if set(frame.dataset) != expected_targets or len(frame) != 270 or frame.model_id.nunique() != 54:
        raise ValueError("Expected complete 54 x 5 classification grid")
    for col, value in {"checkpoint_step": 2500, "training_seed": 0, "evaluation_seed": 0,
                       "eval_episode_seed_offset": 0, "episodes": 128, "n_way": 2, "n_shot": 10}.items():
        if not frame[col].eq(value).all():
            raise ValueError(f"Wrong {col}")
    if not frame.task.eq("classification").all() or not frame.architecture.eq("prodigy").all():
        raise ValueError("Wrong classification task/architecture")
    for col in ("accuracy", "f1", "roc_auc"):
        if not np.isfinite(frame[col]).all() or not frame[col].between(0, 1).all():
            raise ValueError(f"Invalid {col}")
    frame = frame.copy()
    frame["source_set"] = frame.sources.map(lambda x: tuple(sorted(ast.literal_eval(x))))
    frame["k"] = frame.source_set.map(len)
    models = frame.drop_duplicates("model_id")
    if models.groupby("k").size().to_dict() != {1: 9, 2: 36, 8: 9}:
        raise ValueError("Wrong singleton/pair/LOO coverage")
    if frame.groupby("model_id").source_set.nunique().ne(1).any() or models.source_set.duplicated().any():
        raise ValueError("Inconsistent or duplicate source-set models")
    universe = set.union(*(set(x) for x in models.source_set))
    for k in (1, 2, 8):
        expected = set(combinations(sorted(universe), k))
        if set(models.loc[models.k.eq(k), "source_set"]) != expected:
            raise ValueError(f"Incomplete source combinations k={k}")
    for _, group in frame.groupby("dataset"):
        for col in ("episode_fingerprint", "n_query", "queries"):
            if group[col].nunique() != 1:
                raise ValueError(f"Within-target drift: {col}")
    if not frame.queries.eq(frame.episodes * frame.n_way * frame.n_query).all():
        raise ValueError("Query count mismatch")
    frame["target"] = frame.dataset.map(key)
    frame["target_seen"] = [t in s for t, s in zip(frame.target, frame.source_set)]
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    args = parser.parse_args()
    data = args.data_dir
    raw = data / "raw"
    cls = validate_cls(pd.read_csv(raw / "classification_long.tsv", sep="\t"))
    files = list(raw.iterdir())
    single_path = ANALYSIS / "graphs/transfer_prediction/similarity_vs_transfer_v2/data/final_core_auc/specialist_cells_three_seed.csv"
    single = pd.read_csv(single_path).query("seed == 0").copy()
    single["target"] = single.target_key
    single["source_set"] = single.source_key.map(lambda s: (s,))
    single["model_id"] = "ss_" + single.source_key
    pair = pd.read_csv(raw / "pair_nm_long.tsv", sep="\t")
    pair["source_set"] = [tuple(sorted((a, b))) for a, b in zip(pair.source_left, pair.source_right)]
    loo = pd.read_csv(raw / "loo_nm_long.tsv", sep="\t")
    loo["target"] = loo.heldout
    loo["source_set"] = loo.training_sources.map(lambda s: tuple(sorted(s.split(","))))
    nm = pd.concat([single, pair, loo], ignore_index=True)
    for frame, rows in ((single, 81), (pair, 324), (loo, 9)):
        if len(frame) != rows or frame[["model_id", "target"]].duplicated().any():
            raise ValueError("Wrong NM coverage")
    for col, expected in (("seed", 0), ("checkpoint_step", 2500), ("episode_count", 512)):
        if not nm[col].eq(expected).all():
            raise ValueError(f"NM protocol mismatch {col}")
    for col in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
        if nm.groupby("target")[col].nunique().ne(1).any():
            raise ValueError(f"NM fingerprint mismatch {col}")
    nm["k"] = nm.source_set.map(len)
    nm["target_seen"] = [t in s for t, s in zip(nm.target, nm.source_set)]
    nm = nm.rename(columns={"roc_auc_ovr_macro": "roc_auc"})
    if not np.isfinite(nm[["accuracy", "roc_auc"]]).all().all():
        raise ValueError("Nonfinite NM outcomes")

    summary = {"classification": {}, "neighbor_matching": {}}
    envelopes = []
    source_tables = []
    for task, frame in (("classification", cls), ("neighbor_matching", nm)):
        subset = frame[frame.k.eq(1)]
        for metric in ("roc_auc", "accuracy"):
            summary[task][metric] = {"singletons": variance_parts(subset, metric)}
            if task == "classification":
                summary[task][metric]["all_models"] = variance_parts(frame, metric)
            source_scores = subset.assign(source=subset.source_set.map(lambda s: s[0]))
            for (source, target), g in source_scores.groupby(["source", "target"]):
                source_tables.append({"task": task, "metric": metric, "source": source,
                                      "target": target, "target_seen": source == target,
                                      "value": float(g[metric].iloc[0])})
            lookup = {(r.source_set[0], r.target): getattr(r, metric) for r in subset.itertuples()}
            for r in frame[frame.k.gt(1)].itertuples():
                values = [lookup[(s, r.target)] for s in r.source_set]
                envelopes.append({"task": task, "metric": metric, "model_id": r.model_id,
                                  "target": r.target, "k": r.k, "target_seen": r.target_seen,
                                  "value": getattr(r, metric), "best_constituent": max(values),
                                  "mean_constituent": np.mean(values),
                                  "minus_best": getattr(r, metric) - max(values)})
    envelope = pd.DataFrame(envelopes)
    envelope.to_csv(data / "mixture_envelope.csv", index=False)
    envelope.groupby(["task", "metric", "k", "target_seen"]).agg(
        cells=("value", "size"), mean_minus_best=("minus_best", "mean"),
        median_minus_best=("minus_best", "median"), mae_to_best=("minus_best", lambda x: np.abs(x).mean()),
        fraction_above_best=("minus_best", lambda x: (x > 0).mean())
    ).reset_index().to_csv(data / "mixture_envelope_summary.csv", index=False)
    pd.DataFrame(source_tables).to_csv(data / "specialist_scores.csv", index=False)
    target_summary = cls.groupby(["target", "k"]).agg(
        models=("model_id", "size"), mean_auc=("roc_auc", "mean"),
        min_auc=("roc_auc", "min"), max_auc=("roc_auc", "max"),
        mean_accuracy=("accuracy", "mean")
    ).reset_index()
    target_summary.to_csv(data / "classification_target_summary.csv", index=False)

    # Matched partner replacements among foreign pairs: same target, same partner,
    # same source count and nominal budget. These are observed seed-0 contrasts.
    replacements = []
    universe = sorted({s for sources in cls.source_set for s in sources})
    for target, frame in cls[cls.k.eq(2)].groupby("target"):
        lookup = {r.source_set: r.roc_auc for r in frame.itertuples()}
        foreign = [s for s in universe if s != target]
        for a, b in combinations(foreign, 2):
            for partner in (s for s in foreign if s not in (a, b)):
                replacements.append({"target": target, "source_a": a, "source_b": b,
                                     "partner": partner, "auc_a_minus_b":
                                     lookup[tuple(sorted((a, partner)))] - lookup[tuple(sorted((b, partner)))]})
    replacements = pd.DataFrame(replacements)
    replacements.to_csv(data / "foreign_matched_replacements.csv", index=False)
    replacements.groupby(["target", "source_a", "source_b"]).agg(
        partners=("partner", "size"), mean_auc_a_minus_b=("auc_a_minus_b", "mean"),
        min_delta=("auc_a_minus_b", "min"), max_delta=("auc_a_minus_b", "max")
    ).reset_index().to_csv(data / "foreign_matched_replacement_summary.csv", index=False)

    # Existing graph descriptors are contextual, not realized training draws.
    graph_path = ANALYSIS / "graphs/structure/graph_divergence/data/graph_divergence_data.json"
    ext_path = ANALYSIS / "graphs/transfer_prediction/similarity_vs_transfer_v2/data/extended_predictors.json"
    graph = json.loads(graph_path.read_text())
    ext = json.loads(ext_path.read_text())
    stats = pd.DataFrame.from_dict(graph["per_graph"], orient="index")
    stats.index = stats.index.map(key)
    cls_s = cls[cls.k.eq(1)].assign(source=lambda x: x.source_set.map(lambda s: s[0]))
    strength = cls_s.groupby("source").roc_auc.mean()
    desc = []
    for name in ("n_nodes", "n_edges", "density", "largest_wcc_frac", "avg_clustering_approx",
                 "feature_effective_dim", "feature_homophily", "missing_bio_rate", "degree_assortativity"):
        x = stats.loc[strength.index, name].astype(float)
        desc.append({"descriptor": name, "n_sources": len(x), "spearman": spearmanr(x, strength).statistic,
                     "scope": "mean over identical five targets; nine donors; exploratory, no causal interpretation"})
    pd.DataFrame(desc).to_csv(data / "descriptor_associations.csv", index=False)
    correlations = []
    for artifact_name, artifact in (("base", graph), ("extended", ext)):
        order = {key(g): i for i, g in enumerate(artifact["graphs"])}
        for metric, matrix in artifact["pairwise"].items():
            matrix = np.asarray(matrix, dtype=float)
            for target, f in cls_s[~cls_s.target_seen].groupby("target"):
                x = np.array([matrix[order[s], order[target]] for s in f.source])
                finite = np.isfinite(x)
                rho = spearmanr(x[finite], f.roc_auc.to_numpy()[finite]).statistic if finite.sum() >= 3 and np.unique(x[finite]).size > 1 else np.nan
                correlations.append({"artifact": artifact_name, "metric": metric, "target": target,
                                     "n_sources": int(finite.sum()), "rho": rho})
    corr = pd.DataFrame(correlations)
    corr.to_csv(data / "classification_similarity_by_target.csv", index=False)
    corr.groupby(["artifact", "metric"]).agg(targets=("rho", "count"), mean_rho=("rho", "mean"),
        min_rho=("rho", "min"), max_rho=("rho", "max")).reset_index().to_csv(
            data / "classification_similarity_summary.csv", index=False)
    summary["scope"] = {"classification_cells": len(cls), "nm_cells": len(nm),
        "training_seeds": [0], "classification_query_counts": cls.groupby("target").queries.first().to_dict(),
        "caveats": ["No new training or evaluation", "No p-values from matrix cells",
                    "LOO NM has only the omitted target, not a full 9x9 grid",
                    "No all-nine CLS reference imported; LOO results are not all-nine minus source effects",
                    "Fingerprint agreement does not prove identical sampled context subgraphs for CLS",
                    "Historical graph metrics are not sampler-weighted measurements",
                    "Singleton and new mixture training revisions differ; nominal recipe is matched"]}
    summary["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in files + [single_path, graph_path, ext_path]}
    (data / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(target_summary.to_string(index=False))
    print("Specialist mean over the same five targets:\n", strength.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()

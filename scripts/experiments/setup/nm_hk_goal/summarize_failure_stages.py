"""Join cached HK input and stage audits into a descriptive failure taxonomy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(d):
    error = ~d.native_correct
    out = np.full(len(d), "native_correct", dtype=object)
    out[error & d.encoded_correct] = "prem_top1_final_loss"
    remain = error & ~d.encoded_correct
    raw = (d.full_mean_all_valid.astype(bool) &
           d.full_mean_strict_win.fillna(False).astype(bool))
    overlap = (d.node_jaccard_all_valid.astype(bool) &
               d.node_jaccard_strict_win.fillna(False).astype(bool))
    out[remain & raw & overlap] = "prem_loss_raw_and_overlap_signal"
    out[remain & ~raw & overlap] = "prem_loss_overlap_only_signal"
    out[remain & raw & ~overlap] = "prem_loss_raw_only_signal"
    out[remain & ~raw & ~overlap] = "unseparated_by_measured_input_summaries"
    return out


def counts(frame, column="category"):
    n = len(frame)
    return {str(k): {"count": int(v), "fraction": float(v/n)}
            for k, v in frame[column].value_counts().items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--stages", type=Path, required=True)
    p.add_argument("--model", default="hk")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--private-rows", type=Path)
    a = p.parse_args()
    geometry = pd.read_csv(a.geometry)
    stages = pd.read_csv(a.stages)
    stages = stages[stages.model == a.model].copy()
    keys = ["episode", "sample", "query", "anchor"]
    assert len(geometry) == len(stages) == 61440
    assert not geometry.duplicated(keys).any() and not stages.duplicated(keys).any()
    d = geometry.merge(stages, on=keys, validate="one_to_one", suffixes=("_input", ""))
    assert len(d) == 61440
    assert (d.hk_correct.astype(bool) == d.native_correct.astype(bool)).all()
    d["native_correct"] = d.native_correct.astype(bool)
    d["encoded_correct"] = d.encoded_correct.astype(bool)
    d["category"] = classify(d)
    d["query_frequency"] = d.groupby("query")["query"].transform("size")
    d["query_repeated_in_episode"] = d.duplicated(["episode", "query"], keep=False)
    errors = d[~d.native_correct].copy()
    assert len(errors) == 50464
    categories = [c for c in d.category.unique() if c != "native_correct"]
    node_weighted = {}
    distinct = {}
    for c in categories:
        per_node = errors.assign(hit=errors.category.eq(c)).groupby("query").hit.mean()
        node_weighted[c] = float(per_node.mean())
        distinct[c] = {"queries_with_at_least_one": int(errors.loc[errors.category.eq(c), "query"].nunique()),
                       "fraction_of_error_queries": float(errors.loc[errors.category.eq(c), "query"].nunique()/errors["query"].nunique())}
    transitions = (d.groupby(["encoded_correct", "native_correct"]).size()
        .rename("count").reset_index())
    transitions["fraction"] = transitions["count"]/len(d)
    overlap = (d.node_jaccard_all_valid.astype(bool) &
               d.node_jaccard_strict_win.fillna(False).astype(bool))
    raw = (d.full_mean_all_valid.astype(bool) &
           d.full_mean_strict_win.fillna(False).astype(bool))
    signal = pd.DataFrame({
        "population": ["all", "native_errors"],
        "n": [len(d), len(errors)],
        "raw_full_strict": [int(raw.sum()), int(raw[~d.native_correct].sum())],
        "overlap_strict": [int(overlap.sum()), int(overlap[~d.native_correct].sum())],
        "either_strict": [int((raw|overlap).sum()), int((raw|overlap)[~d.native_correct].sum())],
    })
    freq = pd.cut(errors.query_frequency, [0,1,4,19,np.inf], labels=["1","2-4","5-19","20+ "])
    by_frequency = {str(k): {"n": len(g), "categories": counts(g)} for k,g in errors.groupby(freq, observed=True)}
    by_episode = errors.groupby("episode").category.value_counts().unstack(fill_value=0)
    result = {
        "schema": 1, "model": a.model, "rows": len(d), "errors": len(errors),
        "accuracy": float(d.native_correct.mean()), "distinct_queries": int(d["query"].nunique()),
        "distinct_error_queries": int(errors["query"].nunique()),
        "taxonomy": "Mutually exclusive descriptive hierarchy: pre-M top1 lost by final decision; otherwise pre-M misses split by strict raw-full-mean and sampled-node-Jaccard true-class wins. A raw win requires all 30 class scores to be defined.",
        "categories_among_errors": counts(errors),
        "node_weighted_categories_among_errors": node_weighted,
        "distinct_query_coverage": distinct,
        "stage_transitions": transitions.to_dict("records"),
        "measured_input_signal": signal.to_dict("records"),
        "repetition": {
            "error_occurrences_from_queries_repeated_in_same_episode": int(errors.query_repeated_in_episode.sum()),
            "fraction": float(errors.query_repeated_in_episode.mean()),
            "category_counts_repeated_in_episode": counts(errors[errors.query_repeated_in_episode]),
            "by_global_query_frequency": by_frequency,
        },
        "episode_variation": {c: {"min": int(by_episode[c].min()), "median": float(by_episode[c].median()),
                                   "max": int(by_episode[c].max())} for c in by_episode},
        "sources": {"geometry": str(a.geometry), "geometry_sha256": sha256(a.geometry),
                    "stages": str(a.stages), "stages_sha256": sha256(a.stages)},
        "interpretation_limits": [
            "The categories describe observed stage rankings and are not causal percentages.",
            "Raw neighborhood mean is lossy; its failure does not establish absence of input evidence.",
            "Sampled-node Jaccard uses global identities that the model does not receive explicitly.",
            "Occurrences, repeated nodes, and 512 episodes are not independent trained-model replications.",
        ],
    }
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if a.private_rows:
        keep = keys + ["native_correct", "encoded_correct", "category", "query_frequency",
            "query_repeated_in_episode", "full_mean_margin", "node_jaccard_margin",
            "encoded_rank", "encoded_margin", "native_rank", "native_margin"]
        d[keep].to_csv(a.private_rows, index=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

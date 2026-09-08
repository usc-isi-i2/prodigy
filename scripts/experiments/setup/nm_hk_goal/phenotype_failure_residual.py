"""Phenotype canonical HK errors using only cached rows and edge views."""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tab(d, column):
    n = len(d)
    return {str(k): {"count": int(v), "fraction": float(v/n)}
            for k, v in d[column].value_counts(dropna=False).items()}


def cross(d, rows, columns):
    z = pd.crosstab(d[rows], d[columns])
    return {str(c): {str(r): int(v) for r,v in values.items()}
            for c,values in z.to_dict().items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classified", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--stages", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--views-receipt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--private-rows", type=Path)
    a = p.parse_args()
    receipt = json.loads(a.views_receipt.read_text())
    assert digest(a.views) == receipt["canonical_views_sha256"]
    c = pd.read_csv(a.classified)
    g = pd.read_csv(a.geometry)
    s = pd.read_csv(a.stages)
    keys = ["episode", "sample", "query", "anchor"]
    extra = ["query_degree", "query_nodes", "query_edges", "query_zero_fraction",
             "true_support_nodes_mean", "true_shared_nodes", "query_center_in_true_support",
             "query_center_in_any_rival_support", "full_mean_all_valid"]
    d = c.merge(g[keys+extra], on=keys, validate="one_to_one")
    keep = keys+["native_correct", "native_prediction", "native_rank", "native_margin",
                 "encoded_correct", "encoded_prediction", "encoded_rank", "encoded_margin"]
    hk = s[s.model.eq("hk")][keep].rename(columns={x:"hk_"+x for x in keep[4:]})
    ukr = s[s.model.eq("ukr")][keep].rename(columns={x:"ukr_"+x for x in keep[4:]})
    d = d.merge(hk, on=keys, validate="one_to_one").merge(ukr, on=keys, validate="one_to_one")
    assert len(d) == 61440 and (d.native_correct == d.hk_native_correct).all()
    d["class_slot"] = (d["sample"] % 210)//7
    anchors = d.drop_duplicates(["episode", "class_slot"]).set_index(["episode", "class_slot"])["anchor"]
    for model in ["hk", "ukr"]:
        ix = pd.MultiIndex.from_arrays([d.episode, d[model+"_native_prediction"]])
        d[model+"_predicted_anchor"] = anchors.loc[ix].to_numpy()
    packed = torch.load(a.views, map_location="cpu", weights_only=False)
    views = {k: {tuple(sorted(map(int, x))) for x in v.T.tolist()}
             for k,v in packed["views"].items()}
    assert not (views["train"] & views["test"] or views["train"] & views["validation"] or
                views["test"] & views["validation"])
    def split(q, pred):
        edge = tuple(sorted((int(q), int(pred))))
        return next((k for k in ["test", "train", "validation"] if edge in views[k]), "absent")
    for model in ["hk", "ukr"]:
        d[model+"_predicted_edge_split"] = [split(q,p) for q,p in
            zip(d["query"], d[model+"_predicted_anchor"])]
    # Query-level behavior is computed on the complete stream before conditioning.
    q = d.groupby("query").agg(occurrences=("query", "size"),
        hk_accuracy=("hk_native_correct", "mean"), ukr_accuracy=("ukr_native_correct", "mean"))
    d = d.join(q, on="query", rsuffix="_query")
    d["hk_query_behavior"] = np.select(
        [d.hk_accuracy.eq(0), d.hk_accuracy.eq(1), d.hk_accuracy.lt(.25), d.hk_accuracy.lt(.75)],
        ["always_wrong", "always_correct", "mostly_wrong", "mixed"], default="mostly_correct")
    class_outcome = d.groupby(["episode", "class_slot"]).hk_native_correct.sum()
    ix = pd.MultiIndex.from_arrays([d.episode, d.class_slot])
    d["hk_class_correct_of_four"] = class_outcome.loc[ix].to_numpy()
    errors = d[~d.hk_native_correct]
    residual = errors[errors.category.eq("unseparated_by_measured_input_summaries")].copy()
    residual["source_outcome"] = np.select([
        residual.ukr_native_correct,
        residual.hk_native_prediction.eq(residual.ukr_native_prediction)],
        ["foreign_final_correct", "both_wrong_same_class"], default="both_wrong_different_class")
    residual["rank_group"] = pd.cut(residual.hk_native_rank, [0,3,10,30],
                                     labels=["rank_2_3", "rank_4_10", "rank_11_30"])
    residual["raw_status"] = np.where(~residual.full_mean_all_valid.astype(bool), "undefined",
        np.where(residual.full_mean_margin < -1e-7, "rival_closer", "tied"))
    hard = (residual.hk_predicted_edge_split.eq("absent") &
            residual.source_outcome.str.startswith("both_wrong") &
            residual.hk_query_behavior.eq("always_wrong") & residual.hk_native_rank.gt(10))
    populations = {"native_correct": d[d.hk_native_correct],
        "localized_errors": errors[~errors.category.eq("unseparated_by_measured_input_summaries")],
        "residual_errors": residual}
    comparison = {}
    for name,z in populations.items():
        comparison[name] = {"n": len(z),
            "median_query_degree": float(z.query_degree.median()),
            "median_query_occurrences": float(z.occurrences.median()),
            "mean_query_nodes": float(z.query_nodes.mean()),
            "mean_fraction_zero_nodes_in_query_subgraph": float(z.query_zero_fraction.mean()),
            "raw_30way_undefined_fraction": float((~z.full_mean_all_valid.astype(bool)).mean()),
            "query_center_in_rival_support_fraction": float(z.query_center_in_any_rival_support.mean()),
            "all_four_class_wrong_fraction": float(z.hk_class_correct_of_four.eq(0).mean()),
            "query_always_wrong_fraction": float(z.hk_query_behavior.eq("always_wrong").mean()),
            "foreign_final_correct_fraction": float(z.ukr_native_correct.mean())}
    result = {
        "schema": 1, "all_rows": len(d), "native_errors": len(errors), "residual": len(residual),
        "residual_fraction_errors": float(len(residual)/len(errors)),
        "residual_tables": {
            "predicted_edge_split": tab(residual, "hk_predicted_edge_split"),
            "source_outcome": tab(residual, "source_outcome"),
            "query_behavior": tab(residual, "hk_query_behavior"),
            "class_correct_of_four": tab(residual, "hk_class_correct_of_four"),
            "true_rank_group": tab(residual, "rank_group"),
            "raw_status": tab(residual, "raw_status"),
            "query_center_in_rival_support": tab(residual, "query_center_in_any_rival_support"),
        },
        "cross_tabs": {
            "edge_by_source": cross(residual, "hk_predicted_edge_split", "source_outcome"),
            "query_behavior_by_source": cross(residual, "hk_query_behavior", "source_outcome"),
            "rank_by_source": cross(residual, "rank_group", "source_outcome"),
        },
        "dominant_hard_core": {
            "definition": "Residual; HK predicts no recorded edge; both models final-wrong; query always HK-wrong; true rank >10.",
            "count": int(hard.sum()), "fraction_residual": float(hard.mean()),
            "distinct_queries": int(residual.loc[hard, "query"].nunique()),
        },
        "population_comparison": comparison,
        "medians": {col: float(residual[col].median()) for col in
            ["query_degree", "occurrences", "query_nodes", "query_edges", "query_zero_fraction",
             "true_support_nodes_mean", "true_shared_nodes", "hk_native_rank"]},
        "hashes": {str(x): digest(x) for x in [a.classified,a.geometry,a.stages,a.views,a.views_receipt]},
        "limits": ["Outcome-conditioned descriptive groups are not causal mechanisms.",
            "Absent means absent from cached train/validation/test edges, not no real-world relationship.",
            "Foreign correctness compares one checkpoint per source, not independent training replicates."],
    }
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if a.private_rows:
        cols = keys+["category", "hk_native_rank", "hk_native_margin", "ukr_native_correct",
            "source_outcome", "hk_predicted_edge_split", "hk_query_behavior",
            "hk_class_correct_of_four", "rank_group", "raw_status"]+extra
        residual[cols].to_csv(a.private_rows,index=False)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__": main()

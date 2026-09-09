"""Descriptive canonical NM failures; local private inputs, aggregate output only.

Usage: python audit_nm_episode_detail.py --input-root /path/to/private/tables
Inputs: ukr.tsv, hk.tsv, draws.csv, ukr_cases.json, hk_cases.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stats(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(x.median()) if len(x) else None)


def features(d):
    return {k: stats(d[k]) for k in ["query_degree", "ambiguous", "frequency",
            "query_zero", "cos_anchor", "cos_true_support"] if k in d}


def analyze(root):
    out = dict(protocol="canonical_nm_episode_detail_v1", input_sha256={}, targets={})
    for p in sorted(root.iterdir()):
        if p.name in {"ukr.tsv", "hk.tsv", "draws.csv", "ukr_cases.json", "hk_cases.json"}:
            out["input_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    draws = pd.read_csv(root / "draws.csv")
    for model, target in [("ukr", "ukr_rus"), ("hk", "cp_hk")]:
        raw = pd.read_csv(root / (model + ".tsv"), sep="\t")
        d = raw[raw.split == "test"].copy()
        assert len(d) == 61440 and d.episode.nunique() == 512
        assert not d.duplicated(["episode", "sample"]).any()
        assert d.groupby(["episode", "anchor"]).size().eq(4).all()
        col = model + "_correct"
        d["ambiguous"] = d.groupby(["episode", "query"]).anchor.transform("nunique") > 1
        d["frequency"] = d.groupby("query")["query"].transform("size")
        r = dict(query_outcomes={str(int(k)): dict(rows=len(s), features=features(s))
                                  for k, s in d.groupby(col)})
        wrong = d[d[col] == 0]
        r["wrong_details"] = dict(
            true_rank=stats(wrong[model + "_true_rank"]),
            true_rank_le3=float(wrong[model + "_true_rank"].le(3).mean()),
            true_rank_gt10=float(wrong[model + "_true_rank"].gt(10).mean()),
            true_probability=stats(wrong[model + "_true_probability"]))
        for name in ["anchor", "support"]:
            margin = wrong[name + "_margin_" + model].dropna()
            r["wrong_details"][name + "_geometry"] = dict(
                valid=len(margin), predicted_more_similar=float(margin.lt(0).mean()),
                true_more_similar=float(margin.gt(0).mean()), margin=stats(margin))
        # Query/true-support similarity thresholds come from validation only.
        v = raw[raw.split == "val"]
        r["validation_similarity_bins"] = {}
        for metric in ["cos_true_support", "cos_anchor"]:
            edges = np.unique(np.r_[-np.inf, v[metric].dropna().quantile([.2,.4,.6,.8]), np.inf])
            bins = pd.cut(d[metric], edges, labels=False)
            r["validation_similarity_bins"][metric] = dict(
                internal_edges=edges[1:-1].tolist(),
                bins=[dict(bin=int(k), rows=len(s), accuracy=float(s[col].mean()))
                      for k, s in d.groupby(bins)], missing=int(bins.isna().sum()))
        r["within_query_geometry"] = {}
        for keys in [["query"], ["query", "anchor"]]:
            matched = {}
            for metric in ["cos_true_support", "cos_anchor"]:
                paired = d.groupby(keys + [col])[metric].mean().unstack(col).dropna()
                matched[metric] = stats(paired[1] - paired[0])
            r["within_query_geometry"]["+".join(keys)] = matched
        # Compare all-wrong/all-correct anchor classes, not nonexistent binary whole episodes.
        counts = d.groupby(["episode", "anchor"])[col].sum()
        d["class_correct"] = pd.MultiIndex.from_frame(d[["episode", "anchor"]]).map(counts)
        r["classes"] = {str(int(k)): dict(classes=len(s)//4, features=features(s))
                         for k, s in d.groupby("class_correct")}
        # Account descriptively for query mix using validation-only node accuracies,
        # shrunk with ten observations toward the validation overall mean.
        vq = v.groupby("query")[col].agg(["sum", "size"])
        expected = (vq["sum"] + 10*v[col].mean()) / (vq["size"] + 10)
        d["val_query_expected"] = d["query"].map(expected).fillna(v[col].mean())
        ep = d.groupby("episode").agg(accuracy=(col,"mean"), expected=("val_query_expected","mean"))
        pred = model + "_pred"
        # Concentration of the 120 predictions among the 30 candidate anchors.
        concentration = d.groupby("episode")[pred].apply(lambda x: x.value_counts().max()/len(x))
        ep["top_prediction_share"] = concentration
        ep["predicted_classes"] = d.groupby("episode")[pred].nunique()
        ordered = ep.sort_values("accuracy", kind="stable")
        r["episodes"] = dict(min_accuracy=float(ep.accuracy.min()), max_accuracy=float(ep.accuracy.max()),
            accuracy_expected_spearman=float(ep.accuracy.corr(ep.expected, method="spearman")),
            accuracy_concentration_spearman=float(ep.accuracy.corr(concentration, method="spearman")))
        for label, e in [("bottom_103", ordered.head(103)), ("top_103", ordered.tail(103))]:
            s = d[d.episode.isin(e.index)]
            r["episodes"][label] = dict(episodes=len(e), accuracy=float(e.accuracy.mean()),
                validation_query_expected=float(e.expected.mean()), features=features(s),
                top_prediction_share=stats(e.top_prediction_share), predicted_classes=stats(e.predicted_classes),
                all_wrong_class_fraction=float(s.groupby(["episode","anchor"])[col].sum().eq(0).mean()))
        cases = pd.DataFrame(json.loads((root / (model + "_cases.json")).read_text()))
        trial = draws[(draws.target == target) & (draws.model == model)]
        assert len(trial) == 2000 and trial.groupby(["case","condition"]).size().eq(5).all()
        a = trial[trial.condition == "alternative"].groupby("case").correct.agg(["mean","max"])
        c = trial[trial.condition == "same_context"].groupby("case").correct.max()
        cases = cases.set_index("case").join(a).join(c.rename("context_rescued"))
        r["intervention_groups"] = {}
        for cohort in ["native_failed", "native_correct"]:
            selected = cases[cases.cohort == cohort]
            group = selected["max"] if cohort == "native_failed" else selected["mean"].lt(1).astype(int)
            r["intervention_groups"][cohort] = {
                str(int(k)): dict(cases=len(s), features=features(s),
                    baseline_rank=stats(s[model+"_true_rank"]),
                    alternative_pool_size=stats(s.alternative_pool_size),
                    context_any_correct=float(s.context_rescued.mean()))
                for k,s in selected.groupby(group)}
        out["targets"][target] = r
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(analyze(args.input_root), indent=2, allow_nan=False))

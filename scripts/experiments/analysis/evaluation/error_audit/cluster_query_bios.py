#!/usr/bin/env python3
"""Outcome-blind query embedding clusters, with held-stream error diagnostics.

Run on Tucker. Private files contain bios/ids; only aggregate JSON is portable.
Fit on original unique query nodes; assign fresh nodes without refitting.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


def metrics(df):
    u, h = df.ukr_correct.to_numpy(), df.hk_correct.to_numpy()
    return dict(n=len(df), unique_nodes=int(df.query_node_id.nunique()),
                ukr_error=float(1-u.mean()), hk_error=float(1-h.mean()),
                gap_ukr_accuracy_minus_hk=float((u-h).mean()),
                both_wrong=float(((u == 0) & (h == 0)).mean()),
                ukr_only=float(((u == 1) & (h == 0)).mean()),
                hk_only=float(((u == 0) & (h == 1)).mean()),
                positive_fraction=float(df.true_label.mean()),
                isolation_fraction=float(df.full_isolated.mean()))


def conditional(df):
    # Residual paired accuracy gap relative to stream-wide label x isolation cells.
    delta = df.ukr_correct-df.hk_correct
    expected = delta.groupby([df.true_label, df.full_isolated]).transform("mean")
    out = df.copy()
    out["residual_gap"] = delta-expected
    out["paired_gap"] = delta
    return out


def bootstrap_gap(df, seed):
    # Resample whole episodes to respect dependence among their query occurrences.
    e = df.groupby("episode_id").paired_gap.agg(["sum", "size"]).to_numpy()
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(e), size=(1000, len(e)))
    rates = e[ix, 0].sum(axis=1)/e[ix, 1].sum(axis=1)
    return np.quantile(rates, [.025, .975]).tolist()


def clean_text(s):
    return re.sub(r"https?://\S+|@\w+", " ", str(s))


def run(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.rows, sep="\t", keep_default_na=False)
    assert set(df.stream) == {"original", "fresh"}
    for c in ["query_node_id", "true_label", "ukr_correct", "hk_correct", "full_isolated"]:
        df[c] = pd.to_numeric(df[c])
    for c in ["true_label", "ukr_correct", "hk_correct", "full_isolated"]:
        assert set(df[c]) <= {0, 1}, c
    nodes = df.sort_values(["stream", "episode_id"]).drop_duplicates("query_node_id").sort_values("query_node_id")
    ids = nodes.query_node_id.to_numpy()
    original_ids = set(df.loc[df.stream == "original", "query_node_id"])
    fit_mask = np.array([n in original_ids for n in ids])
    g = torch.load(args.graph, map_location="cpu", weights_only=False)
    x = g["x"] if isinstance(g, dict) else g.x
    x = x[torch.as_tensor(ids)].float().numpy()
    assert np.isfinite(x).all()
    norms = np.linalg.norm(x, axis=1)
    assert (norms > 0).all(), "Zero vectors require an explicit missing-feature group"
    x = normalize(x)
    y = g["y"] if isinstance(g, dict) else g.y
    assert np.array_equal(y[torch.as_tensor(ids)].numpy().reshape(-1), nodes.true_label.to_numpy())
    del g
    sha = hashlib.sha256(args.rows.read_bytes()).hexdigest()
    result = dict(protocol={
        "rows": str(args.rows), "rows_sha256": sha, "graph": str(args.graph),
        "fit": "original unique query nodes only; fresh assigned with frozen centroids",
        "algorithm": "KMeans on L2-normalized GTE vectors, Euclidean assignment; 20 restarts",
        "primary_k": args.k, "seed": args.seed, "dimensions": x.shape[1],
        "unique_nodes": len(ids), "fit_nodes": int(fit_mask.sum()),
        "fresh_only_nodes": int((~fit_mask).sum()),
        "overlap_nodes": len(original_ids & set(df.loc[df.stream == "fresh", "query_node_id"])),
        "raw_norm_quantiles": np.quantile(norms, [0, .5, 1]).tolist(),
        "caveat": "Descriptive exploratory partition; fresh stream is not independent training or disjoint nodes.",
    }, baseline={}, runs={})
    for s in ["original", "fresh"]:
        result["baseline"][s] = metrics(df[df.stream == s])
    # Lexical labels use original text only and never use model correctness.
    vect = TfidfVectorizer(stop_words="english", min_df=5, max_df=.65,
                          ngram_range=(1, 2), max_features=10000,
                          token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")
    tf = vect.fit_transform(nodes.loc[fit_mask, "bio_text"].map(clean_text))
    words = vect.get_feature_names_out()
    main_assign = None
    for k in sorted(set([4, args.k, 12])):
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=20).fit(x[fit_mask])
        labels = km.predict(x)
        fit_labels = labels[fit_mask]
        alternate = KMeans(n_clusters=k, random_state=args.seed+1, n_init=20).fit_predict(x[fit_mask])
        run_summary = dict(
            silhouette_cosine=float(silhouette_score(x[fit_mask], fit_labels, metric="cosine", sample_size=min(2000, fit_mask.sum()), random_state=args.seed)),
            seed_adjusted_rand=float(adjusted_rand_score(fit_labels, alternate)), clusters=[])
        assigned = df.copy()
        assigned["cluster"] = assigned.query_node_id.map(dict(zip(ids, labels)))
        streams = {s: conditional(assigned[assigned.stream == s]) for s in ["original", "fresh"]}
        reps = []
        for c in range(k):
            in_fit = fit_labels == c
            means = np.asarray(tf[in_fit].mean(axis=0)).ravel()
            other = np.asarray(tf[~in_fit].mean(axis=0)).ravel()
            distinctive = (means-other).argsort()[::-1][:12]
            terms = words[distinctive].tolist()
            info = dict(cluster=c, original_unique_nodes=int(in_fit.sum()), terms=terms, streams={})
            for s, sdf in streams.items():
                group = sdf[sdf.cluster == c]
                info["streams"][s] = metrics(group)
                info["streams"][s]["gap_episode_bootstrap_95"] = bootstrap_gap(group, args.seed)
                info["streams"][s]["gap_residual_label_isolation"] = float(group.residual_gap.mean())
                info["streams"][s]["by_label"] = {str(l): metrics(z) for l, z in group.groupby("true_label")}
                novel = group[~group.query_node_id.isin(original_ids)]
                if s == "fresh" and len(novel):
                    info["streams"][s]["novel_nodes_only"] = metrics(novel)
            selected = np.flatnonzero(labels == c)
            distances = np.linalg.norm(x[selected]-km.cluster_centers_[c], axis=1)
            ordered = selected[np.argsort(distances)]
            for idx in ordered[:6]:
                reps.append(dict(cluster=c, selection="nearest_centroid", query_node_id=int(ids[idx]), bio_text=nodes.iloc[idx].bio_text))
            rng = np.random.default_rng(args.seed+c)
            for idx in rng.choice(selected, size=min(4, len(selected)), replace=False):
                reps.append(dict(cluster=c, selection="random", query_node_id=int(ids[idx]), bio_text=nodes.iloc[idx].bio_text))
            run_summary["clusters"].append(info)
        gaps = np.array([[c["streams"][s]["gap_ukr_accuracy_minus_hk"] for s in ["original", "fresh"]] for c in run_summary["clusters"]])
        run_summary["gap_original_fresh_correlation"] = float(np.corrcoef(gaps.T)[0, 1])
        result["runs"][str(k)] = run_summary
        if k == args.k:
            main_assign = assigned
            assigned.to_csv(out/"query_clusters_private.tsv", sep="\t", index=False)
            pd.DataFrame(reps).to_csv(out/"representative_bios_private.tsv", sep="\t", index=False)
            np.savez_compressed(out/"cluster_centroids.npz", centroids=km.cluster_centers_)
        print(f"k={k}: silhouette={run_summary['silhouette_cosine']:.3f}, seed ARI={run_summary['seed_adjusted_rand']:.3f}, stream gap r={run_summary['gap_original_fresh_correlation']:.3f}", flush=True)
    (out/"query_cluster_summary.json").write_text(json.dumps(result, indent=2))
    assert main_assign is not None and len(main_assign) == len(df)
    print("Completed", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args)

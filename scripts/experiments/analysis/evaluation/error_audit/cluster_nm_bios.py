#!/usr/bin/env python3
"""NM query bio clusters and query/anchor/support cosine error diagnostics.

CPU-only, reads existing predictions and graphs. Fit on validation unique nodes,
assign test with frozen centroids; raw bios/ids remain in the private output dir.
Run in bio-embeddings-v001 (torch/sklearn/duckdb); no model inference is performed.
"""
import argparse
import gc
import json
from itertools import zip_longest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from cluster_query_bios import clean_text


def read_pairs(paths, split):
    rows = []
    with open(paths["ukr"]) as fu, open(paths["hk"]) as fh:
        for line_u, line_h in zip_longest(fu, fh):
            assert line_u and line_h, "Unpaired file lengths"
            u, h = json.loads(line_u), json.loads(line_h)
            for key in ["dataset", "split", "batch_index", "task_id", "sample_index",
                        "query_node_id", "gt", "gt_local", "episode_label_map"]:
                assert u[key] == h[key], (len(rows), key)
            assert u["split"] == split
            assert len(u["episode_label_map"]) == len(u["probabilities"]) == len(h["probabilities"]) == 30
            def supports(r, local):
                return [s["node_id"] for s in r["supports"] if s["local_label"] == local]
            assert supports(u, u["gt_local"]) == supports(h, h["gt_local"])
            assert len(supports(u, u["gt_local"])) == 3
            record = dict(split=split, episode=u["batch_index"], sample=u["sample_index"],
                          query=u["query_node_id"], anchor=u["gt"],
                          ukr_pred=u["prediction"], hk_pred=h["prediction"],
                          ukr_correct=int(u["correct"]), hk_correct=int(h["correct"]))
            assert record["ukr_correct"] == int(u["prediction"] == u["gt"])
            assert record["hk_correct"] == int(h["prediction"] == h["gt"])
            for prefix, r, local in [("true",u,u["gt_local"]), ("ukr",u,u["pred_local"]), ("hk",h,h["pred_local"])]:
                for j, node in enumerate(supports(r, local)):
                    record[f"{prefix}_support_{j}"] = node
            rows.append(record)
    df = pd.DataFrame(rows)
    assert not df.duplicated(["episode", "sample", "query"]).any()
    return df


def stats(df):
    if not len(df):
        return {"n":0}
    u, h = df.ukr_correct.to_numpy(), df.hk_correct.to_numpy()
    return dict(n=len(df), unique_queries=int(df["query"].nunique()),
                ukr_accuracy=float(u.mean()), hk_accuracy=float(h.mean()),
                gap=float((u-h).mean()), both_wrong=float(((u==0)&(h==0)).mean()),
                both_correct=float(((u==1)&(h==1)).mean()),
                ukr_only=float(((u==1)&(h==0)).mean()), hk_only=float(((u==0)&(h==1)).mean()),
                query_zero_fraction=float(df.query_zero.mean()),
                query_degree_median=float(df.query_degree.median()))


def distribution(s):
    s = pd.Series(s).dropna()
    return dict(n=len(s), mean=float(s.mean()) if len(s) else None,
                quantiles={str(p):float(s.quantile(p)) for p in [.1,.25,.5,.75,.9]} if len(s) else {})


def summarize_weighting(args):
    """Reuse saved rows: audit query repetition and validation-selected routing."""
    out = args.out_dir/args.target
    df = pd.read_parquet(out/"paired_cluster_queries_private.parquet")
    summary = {"splits": {}}
    val_gap = df[df.split=="val"].groupby("cluster").delta.mean()
    route_ukr = set(int(c) for c,gap in val_gap.items() if gap>0)
    summary["routing_ukr_clusters_selected_on_val"] = sorted(route_ukr)
    for split in ["val", "test"]:
        sub = df[df.split==split].copy()
        pernode = sub.groupby("query").agg(occurrences=("delta","size"),
            ukr_accuracy=("ukr_correct","mean"),hk_accuracy=("hk_correct","mean"),
            degree=("query_degree","first"))
        pernode["frequency_bin"] = pd.cut(pernode.occurrences,[0,1,4,19,float("inf")],labels=["1","2-4","5-19","20+"])
        sub["frequency_bin"] = sub["query"].map(pernode.frequency_bin)
        selected = np.where(sub.cluster.isin(route_ukr),sub.ukr_correct,sub.hk_correct)
        anchors_per_query = sub.groupby(["episode","query"]).anchor.nunique().rename("distinct_anchors")
        sub = sub.join(anchors_per_query,on=["episode","query"])
        center_consistent_max = sub.groupby(["episode","query","anchor"]).size().groupby(level=[0,1]).max().sum()/len(sub)
        s=dict(node_weighted_ukr_accuracy=float(pernode.ukr_accuracy.mean()),
            node_weighted_hk_accuracy=float(pernode.hk_accuracy.mean()),
            cluster_router_accuracy=float(selected.mean()),
            top_one_percent_nodes_occurrence_share=float(pernode.occurrences.nlargest(max(1,int(np.ceil(len(pernode)*.01)))).sum()/len(sub)),
            max_occurrences_per_query=int(pernode.occurrences.max()),
            multi_anchor_query_occurrence_fraction=float((sub.distinct_anchors>1).mean()),
            max_anchors_for_same_query_in_episode=int(anchors_per_query.max()),
            center_consistent_oracle_accuracy=float(center_consistent_max),
            anchor_ambiguity_groups={str(amb):stats(z) for amb,z in sub.groupby(sub.distinct_anchors>1)},
            frequency_groups={})
        for group,z in sub.groupby("frequency_bin",observed=True):
            s["frequency_groups"][str(group)] = stats(z)
        s["novel_vs_val"] = stats(sub[~sub["query"].isin(df.loc[df.split=="val","query"])])
        summary["splits"][split] = s
    (out/"nm_query_weighting_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def bootstrap(df, seed):
    ep = df.groupby("episode").delta.agg(["sum", "size"]).to_numpy()
    rng = np.random.default_rng(seed)
    i = rng.integers(len(ep), size=(500, len(ep)))
    return np.quantile(ep[i,0].sum(axis=1)/ep[i,1].sum(axis=1), [.025,.975]).tolist()


def load_bios(target, ids, users, data_root, threads):
    selected = pd.DataFrame({"node_id":ids, "userid":[str(users[int(n)]) for n in ids]})
    if target == "cp_hk_twitter":
        d = pd.read_parquet(data_root/target/"parquet/user_bios.parquet")
        d = d.set_index("node_id").reindex(ids)
        assert d.user_id.astype(str).tolist() == selected.userid.tolist()
        return dict(zip(ids, d.profile.fillna("").astype(str)))
    root = data_root/target/"bio_embeddings/gte-multilingual-base/version=v001"
    conn = duckdb.connect(config={"threads":threads})
    conn.register("selected", selected)
    records = conn.execute("""
        WITH candidates AS (
          SELECT s.node_id, o.bio_hash,
                 row_number() OVER(PARTITION BY s.node_id
                   ORDER BY COALESCE(o.last_seen_at,o.first_seen_at) DESC NULLS LAST,
                            o.bio_hash DESC) AS rn
          FROM read_parquet(?) o JOIN selected s ON CAST(o.userid AS VARCHAR)=s.userid
          WHERE o.bio_hash IS NOT NULL
        )
        SELECT c.node_id, b.normalized_bio_text
        FROM candidates c JOIN read_parquet(?) b ON c.bio_hash=b.bio_hash WHERE rn=1
        """, [str(root/"user_bio_observations.parquet"), str(root/"bio_texts.parquet")]).fetchall()
    conn.close()
    return {int(n): str(t) if t is not None else "" for n,t in records}


def analyze(args):
    out = args.out_dir/args.target
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.audit_dir/"nm_ukr_vs_cp_hk_test_summary.json").read_text())[args.target]["paths"]
    paths = {s:{m:p.replace("predictions_test_",f"predictions_{s}_") for m,p in source.items()} for s in ["val","test"]}
    frames = []
    for s in ["val","test"]:
        print(args.target,"pairing",s,flush=True)
        frames.append(read_pairs(paths[s],s))
    df = pd.concat(frames, ignore_index=True)
    id_cols = ["query","anchor","ukr_pred","hk_pred"]+[f"{p}_support_{i}" for p in ["true","ukr","hk"] for i in range(3)]
    assert not df[id_cols].isna().any().any()
    ids = np.unique(df[id_cols].to_numpy().ravel()).astype(np.int64)
    catalog = json.loads(args.catalog.read_text())
    entry = next(g for g in catalog["graphs"] if g["dataset_key"] == args.target)
    root = Path(catalog["data_root"])
    graph_path = root/entry["relative_path"]
    print(args.target,"loading graph and selecting",len(ids),"node vectors",flush=True)
    g = torch.load(graph_path,map_location="cpu",weights_only=False,mmap=True)
    x = g["x"][torch.tensor(ids)].float().numpy()
    assert np.isfinite(x).all()
    norms = np.linalg.norm(x,axis=1)
    x = normalize(x)
    num_nodes = g["x"].shape[0]
    degree = torch.bincount(g["edge_index"].reshape(-1),minlength=num_nodes).numpy()
    user_ids = g["user_ids"]
    del g
    gc.collect()
    node_index = {int(n):i for i,n in enumerate(ids)}
    qids = np.sort(df["query"].unique())
    qi = np.searchsorted(ids,qids)
    qx = x[qi]
    qvalid = norms[qi]>0
    val_ids = set(df.loc[df.split=="val","query"])
    fit_mask = np.array([n in val_ids for n in qids]) & qvalid
    rng = np.random.default_rng(args.seed)
    fit_ix = np.flatnonzero(fit_mask)
    if len(fit_ix)>args.fit_cap:
        fit_ix = np.sort(rng.choice(fit_ix,args.fit_cap,replace=False))
    df["query_zero"] = df["query"].map(lambda n: norms[node_index[n]]==0)
    df["query_degree"] = degree[df["query"].to_numpy()]
    df["degree_bin"] = np.digitize(df.query_degree,[1,2,4,8,16,32,64])
    del degree
    df["delta"] = df.ukr_correct-df.hk_correct
    df["cohort"] = np.select([
        (df.ukr_correct==1)&(df.hk_correct==1),
        (df.ukr_correct==1)&(df.hk_correct==0),
        (df.ukr_correct==0)&(df.hk_correct==1)],
        ["both_correct","ukr_only","hk_only"],default="both_wrong")

    def cosine(c1,c2):
        ia = np.searchsorted(ids,df[c1].to_numpy())
        ib = np.searchsorted(ids,df[c2].to_numpy())
        scores = np.empty(len(df),dtype=np.float32)
        for start in range(0,len(df),4096):
            sl = slice(start,start+4096)
            scores[sl] = np.einsum("ij,ij->i",x[ia[sl]],x[ib[sl]])
        scores[(norms[ia]==0)|(norms[ib]==0)] = np.nan
        return scores
    for column in ["anchor","ukr_pred","hk_pred"]:
        df[f"cos_{column}"] = cosine("query",column)
    for prefix in ["true","ukr","hk"]:
        values = np.column_stack([cosine("query",f"{prefix}_support_{j}") for j in range(3)])
        # Keep fixed three-support denominator; missing inputs are not silently averaged away.
        df[f"cos_{prefix}_support"] = values.mean(axis=1)
    df["anchor_margin_ukr"] = df.cos_anchor-df.cos_ukr_pred
    df["anchor_margin_hk"] = df.cos_anchor-df.cos_hk_pred
    for prefix in ["ukr","hk"]:
        df[f"support_margin_{prefix}"] = df.cos_true_support-df[f"cos_{prefix}_support"]
    relcols = [c for c in df if c.startswith("cos_") or "margin_" in c]
    result = dict(target=entry["canonical_name"],protocol=dict(paths=paths,graph=str(graph_path),
        unique_queries=len(qids),validation_unique_queries=len(val_ids),
        fit_unique_nonzero_nodes=len(fit_ix),fit_cap=args.fit_cap,seed=args.seed,
        zero_query_nodes=int((~qvalid).sum()),dimensions=x.shape[1],
        algorithm="MiniBatchKMeans on L2-normalized GTE; fit validation unique nodes only",
        primary_k=8,bio_selection="latest observed overall for Ukraine; staged row-aligned profile for Hong Kong"),
        baseline={},geometry={},runs={})
    for split in ["val","test"]:
        part = df[df.split==split]
        result["baseline"][split] = stats(part)
        result["baseline"][split]["distinct_episodes"] = int(part.episode.nunique())
        result["baseline"][split]["queries_per_episode"] = sorted(int(n) for n in part.groupby("episode").size().unique())
        result["geometry"][split] = {c:{key:distribution(z[key]) for key in relcols} for c,z in part.groupby("cohort")}
    primary = None
    representative_ids = set()
    rep_records = []
    q_assignments = pd.DataFrame({"query":qids,"zero":~qvalid})
    for k in [4,8,12]:
        print(args.target,"clustering",k,flush=True)
        params = dict(n_clusters=k,n_init=10,batch_size=1024,max_iter=150,max_no_improvement=30,reassignment_ratio=.01)
        km = MiniBatchKMeans(random_state=args.seed,**params).fit(qx[fit_ix])
        labels = np.full(len(qids),-1,dtype=int)
        labels[qvalid] = km.predict(qx[qvalid])
        km2 = MiniBatchKMeans(random_state=args.seed+1,**params).fit(qx[fit_ix])
        summary = dict(seed_ari=float(adjusted_rand_score(labels[fit_ix],km2.predict(qx[fit_ix]))),
            silhouette_cosine=float(silhouette_score(qx[fit_ix],labels[fit_ix],metric="cosine",sample_size=min(2000,len(fit_ix)),random_state=args.seed)),clusters=[])
        labelmap = dict(zip(qids,labels))
        df["cluster"] = df["query"].map(labelmap)
        df["degree_residual_gap"] = df.delta-df.groupby(["split","degree_bin","query_zero"]).delta.transform("mean")
        for c in sorted(set(labels)):
            info = dict(cluster=int(c),unique_queries=int((labels==c).sum()),streams={})
            for s in ["val","test"]:
                sub = df[(df.cluster==c)&(df.split==s)]
                info["streams"][s] = stats(sub)
                if len(sub):
                    info["streams"][s]["gap_episode_bootstrap_95"] = bootstrap(sub,args.seed)
                    info["streams"][s]["degree_residual_gap"] = float(sub.degree_residual_gap.mean())
                    info["streams"][s]["geometry"] = {col:distribution(sub[col]) for col in ["cos_anchor","cos_true_support"]}
                    # Unique-node weighting checks high-frequency query hubs.
                    unique = sub.groupby("query")[["ukr_correct","hk_correct"]].mean()
                    info["streams"][s]["node_weighted_gap"] = float((unique.ukr_correct-unique.hk_correct).mean())
                    if s=="test":
                        info["streams"][s]["unseen_in_validation"] = stats(sub[~sub["query"].isin(val_ids)])
            summary["clusters"].append(info)
            if k==8:
                ii = np.flatnonzero((labels==c)&np.array([n in val_ids for n in qids]))
                if c>=0:
                    distances = np.linalg.norm(qx[ii]-km.cluster_centers_[c],axis=1)
                    near = ii[np.argsort(distances)[:5]]
                else:
                    near = ii[:5]
                random = rng.choice(ii,min(4,len(ii)),replace=False)
                for kind, indices in [("nearest",near),("random",random)]:
                    for index in indices:
                        representative_ids.add(int(qids[index]))
                        rep_records.append(dict(cluster=int(c),selection=kind,node_id=int(qids[index])))
        nonzero = [c for c in summary["clusters"] if c["cluster"]>=0]
        summary["gap_val_test_correlation"] = float(np.corrcoef([[c["streams"][s]["gap"] for c in nonzero] for s in ["val","test"]])[0,1])
        result["runs"][str(k)] = summary
        if k==8:
            primary = labels.copy()
            q_assignments["cluster"] = labels
            df.to_parquet(out/"paired_cluster_queries_private.parquet",index=False)
            np.savez_compressed(out/"cluster_centroids.npz",centroids=km.cluster_centers_)
    # Select actual paired examples before reading text: random cases per test cohort.
    examples=[]
    for cohort,z in df[df.split=="test"].groupby("cohort"):
        for _,r in z.sample(min(8,len(z)),random_state=args.seed).iterrows():
            ex={key:r[key].item() if hasattr(r[key],"item") else r[key] for key in ["query","anchor","ukr_pred","hk_pred","ukr_correct","hk_correct","episode","sample"]+relcols}
            ex["cohort"]=cohort
            ex["cluster"]=int(primary[np.searchsorted(qids,r["query"])] )
            examples.append(ex)
            representative_ids.update(int(ex[key]) for key in ["query","anchor","ukr_pred","hk_pred"])
    # Add sampled original-fit query text for outcome-blind lexical interpretation.
    lexical_ids = set(int(qids[i]) for i in fit_ix)
    text_ids = np.array(sorted(lexical_ids|representative_ids))
    print(args.target,"joining bios for",len(text_ids),"nodes",flush=True)
    bios = load_bios(args.target,text_ids,user_ids,root,args.threads)
    result["protocol"]["text_lookup_nodes"] = len(text_ids)
    result["protocol"]["nonempty_bios_recovered"] = sum(bool(bios.get(int(n),"")) for n in text_ids)
    del user_ids
    for r in rep_records:
        r["bio"] = bios.get(r["node_id"],"")
    pd.DataFrame(rep_records).to_csv(out/"representative_bios_private.tsv",sep="\t",index=False)
    for ex in examples:
        for key in ["query","anchor","ukr_pred","hk_pred"]:
            ex[key+"_bio"] = bios.get(ex[key],"")
    (out/"paired_examples_private.json").write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    docs = [clean_text(bios.get(int(qids[i]),"")) for i in fit_ix]
    vect = TfidfVectorizer(stop_words="english",min_df=5,max_df=.7,max_features=12000,ngram_range=(1,2),token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")
    tf=vect.fit_transform(docs)
    words=vect.get_feature_names_out()
    for c in result["runs"]["8"]["clusters"]:
        label=c["cluster"]
        if label<0:
            c["terms"]=["zero feature vector"]
            continue
        mask=primary[fit_ix]==label
        diff=np.asarray(tf[mask].mean(axis=0)-tf[~mask].mean(axis=0)).ravel()
        c["terms"]=words[diff.argsort()[::-1][:14]].tolist()
    q_assignments.to_csv(out/"unique_query_clusters_private.tsv",sep="\t",index=False)
    (out/"nm_bio_cluster_summary.json").write_text(json.dumps(result,indent=2))
    print(args.target,"DONE",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target",required=True,choices=["ukr_rus_twitter","cp_hk_twitter"])
    ap.add_argument("--audit-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--catalog",type=Path,default=Path("docs/graph_catalog.json"))
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--fit-cap",type=int,default=20000)
    ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--summarize-weighting",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        if args.summarize_weighting:
            summarize_weighting(args)
        else:
            analyze(args)

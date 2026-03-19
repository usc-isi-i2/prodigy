import argparse
import csv
import glob
import os
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


# Reads the midterm CSV format where one logical record may be split across two rows.
def load_interleaved_csv(filepath: str) -> pd.DataFrame:
    main_rows, sub_rows = [], []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            sub_header_raw = next(reader)
        except StopIteration:
            return pd.DataFrame()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        if sub_header_raw is not None:
            next(reader, None)

        pending_main = None
        for row in reader:
            if not row:
                continue
            if len(row) == 66:
                if pending_main is not None:
                    main_rows.append(pending_main)
                    sub_rows.append([""] * 11)
                pending_main = row
            elif len(row) == 11:
                if pending_main is not None:
                    main_rows.append(pending_main)
                    sub_rows.append(row)
                    pending_main = None
            else:
                # malformed line: skip
                continue

        if pending_main is not None:
            main_rows.append(pending_main)
            sub_rows.append([""] * 11)

    sub_cols = [
        "sub_extra", "state", "country", "rt_state", "rt_country",
        "qtd_state", "qtd_country", "norm_country", "norm_rt_country",
        "norm_qtd_country", "acc_age",
    ]

    df_main = pd.DataFrame(main_rows, columns=header)
    df_sub = pd.DataFrame(sub_rows, columns=sub_cols).drop(columns=["sub_extra"], errors="ignore")
    return pd.concat([df_main.reset_index(drop=True), df_sub.reset_index(drop=True)], axis=1)


def count_list_like(val) -> int:
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s in {"", "[]", "nan", "None"}:
        return 0
    s = s.strip("[]")
    if not s:
        return 0
    return len([x for x in s.split(",") if x.strip()])


def parse_args():
    p = argparse.ArgumentParser(description="Generate full-node retweet graph with edge features from raw midterm CSVs.")
    p.add_argument("--csv_glob", default="/project2/ll_774_951/midterm/*/*.csv")
    p.add_argument("--out_dir", default="midterm/graph_retweet_full")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--strict_dates", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    usecols = {
        "userid", "rt_userid", "date", "state",
        "followers_count", "verified", "statuses_count", "sent_vader",
        "hashtag", "mentionsn", "media_urls",
        "rt_fav_count", "rt_reply_count",
    }

    files = sorted(glob.glob(args.csv_glob))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.csv_glob}")

    print(f"Found {len(files)} files")
    t0 = time.time()
    chunks: List[pd.DataFrame] = []

    for i, fpath in enumerate(files, start=1):
        try:
            dfi = load_interleaved_csv(fpath)
            if dfi.empty:
                continue
            cols = [c for c in dfi.columns if c in usecols]
            if not cols:
                continue
            chunks.append(dfi[cols].copy())
        except Exception as exc:
            print(f"[WARN] Failed parsing {os.path.basename(fpath)}: {exc}")

        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files in {time.time() - t0:.1f}s")

    if not chunks:
        raise RuntimeError("No valid data loaded from CSV files")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"Loaded rows: {len(df):,}")

    # Normalize IDs
    df["userid"] = pd.to_numeric(df.get("userid"), errors="coerce")
    df["rt_userid"] = pd.to_numeric(df.get("rt_userid"), errors="coerce")

    # Parse timestamps (needed for temporal edge feature)
    df["date"] = df.get("date", "").astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(
        df["date"],
        format="%a %b %d %H:%M:%S +0000 %Y",
        utc=True,
        errors="coerce",
    )
    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        n_bad = int(bad_ts.sum())
        examples = df.loc[bad_ts, "date"].drop_duplicates().head(5).tolist()
        if args.strict_dates:
            raise ValueError(f"Invalid timestamp rows: {n_bad:,}, examples: {examples}")
        print(f"Dropping invalid timestamp rows: {n_bad:,}")
        df = df.loc[~bad_ts].copy()

    # Keep only rows that are actual retweets for edges.
    rt = df.dropna(subset=["userid", "rt_userid"]).copy()
    rt["userid"] = rt["userid"].astype(np.int64)
    rt["rt_userid"] = rt["rt_userid"].astype(np.int64)

    # All nodes = union of all sources and all retweet targets.
    node_ids = np.array(sorted(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist())), dtype=np.int64)
    id_to_idx = {uid: i for i, uid in enumerate(node_ids)}
    n_nodes = len(node_ids)
    print(f"Nodes (userid U rt_userid): {n_nodes:,}")

    # Edge aggregation per directed pair (u -> v)
    edge_grp = rt.groupby(["userid", "rt_userid"], as_index=False).agg(
        first_ts=("timestamp", "min"),
        n_retweets=("timestamp", "size"),
        avg_rt_fav=("rt_fav_count", "mean"),
        avg_rt_reply=("rt_reply_count", "mean"),
    )

    global_min_ts = edge_grp["first_ts"].min()
    min_ns = int(global_min_ts.value)
    first_hours = (edge_grp["first_ts"].astype("int64") - min_ns) / 3_600_000_000_000.0

    edge_grp["first_retweet_time"] = first_hours.astype(float)
    edge_grp["n_retweets"] = np.log1p(edge_grp["n_retweets"].astype(float))
    edge_grp["avg_rt_fav"] = np.log1p(pd.to_numeric(edge_grp["avg_rt_fav"], errors="coerce").fillna(0).clip(lower=0))
    edge_grp["avg_rt_reply"] = np.log1p(pd.to_numeric(edge_grp["avg_rt_reply"], errors="coerce").fillna(0).clip(lower=0))

    edge_grp["src"] = edge_grp["userid"].map(id_to_idx)
    edge_grp["dst"] = edge_grp["rt_userid"].map(id_to_idx)
    edge_grp = edge_grp.dropna(subset=["src", "dst"]).copy()
    edge_grp[["src", "dst"]] = edge_grp[["src", "dst"]].astype(int)

    edge_index = torch.tensor(edge_grp[["src", "dst"]].values.T, dtype=torch.long)

    edge_feature_names = ["first_retweet_time", "n_retweets", "avg_rt_fav", "avg_rt_reply"]
    edge_attr_np = edge_grp[edge_feature_names].fillna(0).values.astype(np.float32)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)

    print(f"Directed unique edges: {edge_index.shape[1]:,}")

    # Node features (for users with authored tweets). Targets-only nodes remain mostly zeros except degree features.
    base = df.dropna(subset=["userid"]).copy()
    base["userid"] = base["userid"].astype(np.int64)

    for col in ["followers_count", "statuses_count", "sent_vader", "rt_fav_count", "rt_reply_count"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    base["verified"] = base.get("verified", 0).map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(float)
    base["n_hashtags"] = base.get("hashtag", "").apply(count_list_like)
    base["n_mentions"] = base.get("mentionsn", "").apply(count_list_like)
    base["has_media"] = base.get("media_urls", "").apply(lambda x: 0 if str(x).strip() in {"", "[]", "nan", "None"} else 1)

    node_agg = base.groupby("userid", as_index=False).agg(
        subscriber_count=("followers_count", "max"),
        verified=("verified", "max"),
        avg_favorites=("rt_fav_count", "mean"),
        avg_comments=("rt_reply_count", "mean"),
        avg_score=("sent_vader", "mean"),
        avg_n_hashtags=("n_hashtags", "mean"),
        avg_n_mentions=("n_mentions", "mean"),
        avg_has_media=("has_media", "mean"),
        post_count=("statuses_count", "max"),
    )

    for col in ["subscriber_count", "avg_favorites", "avg_comments", "post_count"]:
        node_agg[col] = np.log1p(pd.to_numeric(node_agg[col], errors="coerce").fillna(0).clip(lower=0))

    in_deg = edge_grp.groupby("rt_userid").size().rename("in_degree")
    out_deg = edge_grp.groupby("userid").size().rename("out_degree")

    node_agg = node_agg.merge(out_deg, left_on="userid", right_index=True, how="left")
    node_agg = node_agg.merge(in_deg, left_on="userid", right_index=True, how="left")
    node_agg["in_degree"] = np.log1p(node_agg["in_degree"].fillna(0))
    node_agg["out_degree"] = np.log1p(node_agg["out_degree"].fillna(0))

    feature_names = [
        "subscriber_count", "verified", "avg_favorites", "avg_comments", "avg_score",
        "avg_n_hashtags", "avg_n_mentions", "avg_has_media", "post_count", "in_degree", "out_degree",
    ]

    X = np.zeros((n_nodes, len(feature_names)), dtype=np.float32)
    node_agg["node_idx"] = node_agg["userid"].map(id_to_idx)
    has_idx = node_agg["node_idx"].notna()
    rows = node_agg.loc[has_idx, "node_idx"].astype(int).values
    X[rows] = node_agg.loc[has_idx, feature_names].fillna(0).values.astype(np.float32)

    # Fill degree features for pure target nodes too.
    all_out = np.zeros(n_nodes, dtype=np.float32)
    all_in = np.zeros(n_nodes, dtype=np.float32)
    for uid, c in edge_grp.groupby("userid").size().items():
        all_out[id_to_idx[uid]] = np.log1p(float(c))
    for uid, c in edge_grp.groupby("rt_userid").size().items():
        all_in[id_to_idx[uid]] = np.log1p(float(c))
    X[:, feature_names.index("out_degree")] = np.maximum(X[:, feature_names.index("out_degree")], all_out)
    X[:, feature_names.index("in_degree")] = np.maximum(X[:, feature_names.index("in_degree")], all_in)

    # Standardize non-empty rows only.
    nonzero = np.any(X != 0, axis=1)
    if nonzero.any():
        scaler = StandardScaler()
        X[nonzero] = scaler.fit_transform(X[nonzero]).astype(np.float32)

    x = torch.tensor(X, dtype=torch.float)

    # Optional labels from state if available; unknown = -1.
    y = torch.full((n_nodes,), -1, dtype=torch.long)
    label_names: List[str] = []
    if "state" in base.columns:
        st = base[base["state"].notna() & base["state"].astype(str).ne("")][["userid", "state"]].drop_duplicates("userid")
        if len(st):
            label_names = sorted(st["state"].astype(str).unique().tolist())
            st2i = {s: i for i, s in enumerate(label_names)}
            for _, r in st.iterrows():
                uid = int(r["userid"])
                if uid in id_to_idx:
                    y[id_to_idx[uid]] = st2i[str(r["state"])]

    out_path = os.path.join(args.out_dir, "graph_data.pt")
    torch.save(
        {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_attr_feature_names": edge_feature_names,
            "user_ids": node_ids,
            "feature_names": feature_names,
            "y": y,
            "label_names": label_names,
        },
        out_path,
    )

    print("\n=== DONE ===")
    print(f"Saved: {out_path}")
    print(f"x: {tuple(x.shape)}")
    print(f"edge_index: {tuple(edge_index.shape)}")
    print(f"edge_attr: {tuple(edge_attr.shape)} ({edge_feature_names})")
    print(f"labeled nodes: {(y >= 0).sum().item():,} / {n_nodes:,}")


if __name__ == "__main__":
    main()

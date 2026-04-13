import argparse
import glob
import json
import os
import shlex
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_CSV = "/project2/ll_774_951/midterm/*/*.csv"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = "data/data/midterm/embeddings/embeddings_all-MiniLM-L6-v2.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Build per-user (userid) pooled text embeddings from raw midterm CSV files.")
    p.add_argument("--csv_glob", default=DEFAULT_CSV)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=0, help="Cap the embedding artifact to this many unique users (0 = no limit)")
    p.add_argument(
        "--stop_after_max_nodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop reading additional files after a file brings the admitted user count to max_nodes. "
             "Disable with --no-stop_after_max_nodes if you want to keep scanning later files and aggregating posts "
             "for already-admitted users.",
    )
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max_seq_len", type=int, default=64)
    p.add_argument("--fp16", action="store_true", default=True)
    return p.parse_args()


def normalize_user_id(val):
    try:
        return int(val)
    except Exception:
        return None


def build_text(row: pd.Series) -> str:
    """
    Pure retweet (rt_text present) -> use retweeted text.
    Quote tweet or original        -> use own text.
    """
    rt = row.get("rt_text")
    if pd.notna(rt) and str(rt).strip():
        return str(rt).strip()
    own = row.get("text")
    if pd.notna(own) and str(own).strip():
        return str(own).strip()
    return ""


def main():
    args = parse_args()
    t0 = time.time()
    command = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])

    files = sorted(glob.glob(args.csv_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.csv_glob}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_len
    if args.fp16 and args.device.startswith("cuda"):
        model = model.half()
    emb_dim = model.get_sentence_embedding_dimension()

    print(f"Model={args.model} device={args.device} fp16={args.fp16} "
          f"seq_len={args.max_seq_len} batch={args.batch_size} files={len(files)}")
    print(f"Input glob={args.csv_glob}")
    print(f"Output={args.out}")

    uid_to_row = {}
    cap = 1 << 16
    sum_mat = np.zeros((cap, emb_dim), dtype=np.float32)
    cnt_arr = np.zeros(cap, dtype=np.int32)

    def ensure_capacity(n_new):
        nonlocal cap, sum_mat, cnt_arr
        need = len(uid_to_row) + n_new
        if need <= cap:
            return
        while cap < need:
            cap *= 2
        new_sum = np.zeros((cap, emb_dim), dtype=np.float32)
        new_sum[: len(uid_to_row)] = sum_mat[: len(uid_to_row)]
        sum_mat = new_sum
        new_cnt = np.zeros(cap, dtype=np.int32)
        new_cnt[: len(uid_to_row)] = cnt_arr[: len(uid_to_row)]
        cnt_arr = new_cnt

    total_posts = 0
    total_items = 0
    total_missing_uid = 0
    total_empty_text = 0
    total_skip_new_uid = 0

    for i, fpath in enumerate(files, start=1):
        ft = time.time()
        print(f"[{i}/{len(files)}] loading {os.path.basename(fpath)}", flush=True)
        try:
            df = pd.read_csv(fpath, low_memory=False, on_bad_lines="skip")
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {os.path.basename(fpath)}: {e}", flush=True)
            continue
        if df.empty:
            continue

        if "userid" not in df.columns:
            print(f"[{i}/{len(files)}] missing userid, skipping", flush=True)
            continue

        df["userid"] = pd.to_numeric(df["userid"], errors="coerce")
        df = df.dropna(subset=["userid"]).copy()
        df["userid"] = df["userid"].astype(np.int64)
        total_items += len(df)

        texts = []
        row_uids = []
        file_missing_uid = 0
        file_empty_text = 0
        file_skip_new_uid = 0

        for _, r in df.iterrows():
            uid = int(r["userid"])
            text = build_text(r)
            if not text:
                file_empty_text += 1
                continue
            if uid not in uid_to_row:
                if args.max_nodes > 0 and len(uid_to_row) >= args.max_nodes:
                    file_skip_new_uid += 1
                    continue
                ensure_capacity(1)
                uid_to_row[uid] = len(uid_to_row)
            texts.append(text)
            row_uids.append(uid)

        if not texts:
            total_missing_uid += file_missing_uid
            total_empty_text += file_empty_text
            continue

        unique_new = len({u for u in row_uids if uid_to_row.get(u, -1) >= 0})  # overwritten below for logging clarity
        unique_new = len({u for u in row_uids if cnt_arr[uid_to_row[u]] == 0})

        row_idx = np.fromiter((uid_to_row[u] for u in row_uids), dtype=np.int64, count=len(row_uids))

        embs = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

        np.add.at(sum_mat, row_idx, embs)
        np.add.at(cnt_arr, row_idx, 1)

        total_posts += len(texts)
        total_missing_uid += file_missing_uid
        total_empty_text += file_empty_text
        total_skip_new_uid += file_skip_new_uid
        dt = time.time() - ft
        print(
            f"[{i}/{len(files)}] {os.path.basename(fpath)} "
            f"rows={len(df):,} embedded={len(texts):,} "
            f"new_users={unique_new:,} users={len(uid_to_row):,} "
            f"skip_uid={file_missing_uid:,} skip_empty={file_empty_text:,} skip_new_uid={file_skip_new_uid:,} "
            f"file={dt:.1f}s total={(time.time()-t0)/60:.1f}m",
            flush=True,
        )
        if args.stop_after_max_nodes and args.max_nodes > 0 and len(uid_to_row) >= args.max_nodes:
            print(
                f"Reached max_nodes={args.max_nodes:,} after file {i}/{len(files)}; "
                "stopping additional file reads because --stop_after_max_nodes is set.",
                flush=True,
            )
            break

    n = len(uid_to_row)
    user_ids = np.empty(n, dtype=np.int64)
    for uid, row in uid_to_row.items():
        user_ids[row] = uid

    counts_final = cnt_arr[:n].astype(np.int64)
    denom = np.maximum(counts_final, 1).astype(np.float32)[:, None]
    meanpool = sum_mat[:n] / denom
    norms = np.linalg.norm(meanpool, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    meanpool = meanpool / norms

    out_obj = {
        "user_ids": user_ids,
        "meanpool": torch.from_numpy(meanpool),
        "counts": counts_final,
        "model": args.model,
    }
    torch.save(out_obj, args.out)

    meta = {
        "csv_glob": args.csv_glob,
        "files_count": len(files),
        "model": args.model,
        "embedding_dim": int(emb_dim),
        "users": int(n),
        "total_posts_embedded": int(total_posts),
        "max_nodes": int(args.max_nodes),
        "stop_after_max_nodes": bool(args.stop_after_max_nodes),
        "command": command,
        "max_seq_len": args.max_seq_len,
        "fp16": bool(args.fp16),
    }
    with open(args.out.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {args.out} users={n:,} dim={emb_dim} posts={total_posts:,} "
          f"wall={(time.time()-t0)/60:.1f}m")
    print(
        f"Summary: rows_seen={total_items:,} embedded={total_posts:,} "
        f"skip_uid={total_missing_uid:,} skip_empty={total_empty_text:,} skip_new_uid={total_skip_new_uid:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

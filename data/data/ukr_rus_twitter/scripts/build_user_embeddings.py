import argparse
import csv
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


DEFAULT_CSV = "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = "data/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-user pooled text embeddings from raw ukr_rus_twitter CSV files."
    )
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


def load_interleaved_csv(filepath: str) -> pd.DataFrame:
    main_rows, sub_rows = [], []

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            sub_header_raw = next(reader)
        except StopIteration:
            return pd.DataFrame()

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        if sub_header_raw is not None:
            next(reader)

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


def normalize_handle(h):
    if h is None:
        return None
    s = str(h).strip().lower()
    return s if s and s not in {"nan", "none", "<na>"} else None


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


def read_post_file(fpath: str) -> pd.DataFrame:
    try:
        df = load_interleaved_csv(fpath)
        if df.empty:
            return df
    except Exception:
        df = pd.read_csv(fpath, low_memory=False, encoding="utf-8", on_bad_lines="skip")
    df.replace("", pd.NA, inplace=True)
    return df


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

    handle_to_row = {}
    handles = []
    handle_to_userid = {}  # best-effort: screen_name -> userid
    cap = 1 << 16
    sum_mat = np.zeros((cap, emb_dim), dtype=np.float32)
    cnt_arr = np.zeros(cap, dtype=np.int32)

    def ensure_capacity(n_new):
        nonlocal cap, sum_mat, cnt_arr
        need = len(handle_to_row) + n_new
        if need <= cap:
            return
        while cap < need:
            cap *= 2
        new_sum = np.zeros((cap, emb_dim), dtype=np.float32)
        new_sum[: len(handle_to_row)] = sum_mat[: len(handle_to_row)]
        sum_mat = new_sum
        new_cnt = np.zeros(cap, dtype=np.int32)
        new_cnt[: len(handle_to_row)] = cnt_arr[: len(handle_to_row)]
        cnt_arr = new_cnt

    total_posts = 0
    total_items = 0
    total_missing_handle = 0
    total_empty_text = 0
    total_skip_new_handle = 0

    for i, fpath in enumerate(files, start=1):
        ft = time.time()
        print(f"[{i}/{len(files)}] loading {os.path.basename(fpath)}", flush=True)
        try:
            df = read_post_file(fpath)
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {os.path.basename(fpath)}: {e}", flush=True)
            continue
        if df.empty:
            print(f"[{i}/{len(files)}] empty {os.path.basename(fpath)}", flush=True)
            continue

        if "screen_name" not in df.columns:
            print(f"[{i}/{len(files)}] missing screen_name, skipping", flush=True)
            continue

        df["screen_name"] = df["screen_name"].apply(normalize_handle)
        df = df[df["screen_name"].notna()].copy()
        if "userid" in df.columns:
            df["userid"] = pd.to_numeric(df["userid"], errors="coerce")
        total_items += len(df)

        texts = []
        row_handles = []
        file_missing_handle = 0
        file_empty_text = 0
        file_skip_new_handle = 0

        for _, r in df.iterrows():
            handle = r["screen_name"]
            if not handle:
                file_missing_handle += 1
                continue
            text = build_text(r)
            if not text:
                file_empty_text += 1
                continue
            if handle not in handle_to_row:
                if args.max_nodes > 0 and len(handle_to_row) >= args.max_nodes:
                    file_skip_new_handle += 1
                    continue
                ensure_capacity(1)
                handle_to_row[handle] = len(handle_to_row)
                handles.append(handle)
                uid = r.get("userid")
                if pd.notna(uid) and handle not in handle_to_userid:
                    handle_to_userid[handle] = int(uid)
            texts.append(text)
            row_handles.append(handle)

        if not texts:
            total_missing_handle += file_missing_handle
            total_empty_text += file_empty_text
            continue

        unique_new = len({h for h in row_handles if cnt_arr[handle_to_row[h]] == 0})

        row_idx = np.fromiter((handle_to_row[h] for h in row_handles), dtype=np.int64, count=len(row_handles))

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
        total_missing_handle += file_missing_handle
        total_empty_text += file_empty_text
        total_skip_new_handle += file_skip_new_handle
        dt = time.time() - ft
        print(
            f"[{i}/{len(files)}] {os.path.basename(fpath)} "
            f"rows={len(df):,} embedded={len(texts):,} "
            f"new_users={unique_new:,} users={len(handle_to_row):,} "
            f"skip_handle={file_missing_handle:,} skip_empty={file_empty_text:,} skip_new_handle={file_skip_new_handle:,} "
            f"file={dt:.1f}s total={(time.time()-t0)/60:.1f}m",
            flush=True,
        )
        if args.stop_after_max_nodes and args.max_nodes > 0 and len(handle_to_row) >= args.max_nodes:
            print(
                f"Reached max_nodes={args.max_nodes:,} after file {i}/{len(files)}; "
                "stopping additional file reads because --stop_after_max_nodes is set.",
                flush=True,
            )
            break

    n = len(handle_to_row)
    if args.max_nodes > 0:
        if n > args.max_nodes:
            raise RuntimeError(f"Embedding cap failed: requested max_nodes={args.max_nodes:,}, got {n:,}")
        if n < args.max_nodes:
            print(
                f"[WARN] requested max_nodes={args.max_nodes:,}, but only {n:,} users were admitted.",
                flush=True,
            )
    counts_final = cnt_arr[:n].astype(np.int64)
    denom = np.maximum(counts_final, 1).astype(np.float32)[:, None]
    meanpool = sum_mat[:n] / denom
    norms = np.linalg.norm(meanpool, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    meanpool = meanpool / norms

    user_ids = [handle_to_userid.get(h) for h in handles]  # None if userid not found in CSVs

    out_obj = {
        "handles": handles,
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
        f"skip_handle={total_missing_handle:,} skip_empty={total_empty_text:,} "
        f"skip_new_handle={total_skip_new_handle:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

import argparse
import csv
import glob
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_CSV = "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = "data/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt"
TEXT_FIELD_TO_COLUMN = {
    "tweet_text": "text",
    "retweet_text": "rt_text",
    "quote_text": "qtd_text",
    "bio": "description",
}
DEFAULT_TEXT_FIELDS = "tweet_text,retweet_text,quote_text"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-user pooled text embeddings from raw ukr_rus_twitter CSV files."
    )
    p.add_argument("--csv_glob", default=DEFAULT_CSV)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument(
        "--checkpoint_path",
        default="data/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.checkpoint.pkl",
    )
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument(
        "--text_fields",
        default=DEFAULT_TEXT_FIELDS,
        help="Comma-separated semantic text fields to embed. Supported: tweet_text,retweet_text,quote_text,bio",
    )
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
        "sub_extra",
        "state",
        "country",
        "rt_state",
        "rt_country",
        "qtd_state",
        "qtd_country",
        "norm_country",
        "norm_rt_country",
        "norm_qtd_country",
        "acc_age",
    ]

    df_main = pd.DataFrame(main_rows, columns=header)
    df_sub = pd.DataFrame(sub_rows, columns=sub_cols).drop(columns=["sub_extra"], errors="ignore")
    return pd.concat([df_main.reset_index(drop=True), df_sub.reset_index(drop=True)], axis=1)


def normalize_handle(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def parse_text_fields(spec: str):
    fields = [part.strip() for part in str(spec).split(",") if part.strip()]
    if not fields:
        raise ValueError("text_fields must contain at least one field")
    invalid = [field for field in fields if field not in TEXT_FIELD_TO_COLUMN]
    if invalid:
        raise ValueError(
            f"Unsupported text field(s): {invalid}. Supported: {sorted(TEXT_FIELD_TO_COLUMN.keys())}"
        )
    return fields


def build_text(row: pd.Series, text_fields) -> str:
    parts = []
    seen = set()
    for field in text_fields:
        col = TEXT_FIELD_TO_COLUMN[field]
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            value = str(val).strip()
            if value not in seen:
                parts.append(value)
                seen.add(value)
    return " ".join(parts)[:512] if parts else ""


def read_post_file(fpath: str) -> pd.DataFrame:
    try:
        df = load_interleaved_csv(fpath)
        if df.empty:
            return df
    except Exception:
        df = pd.read_csv(fpath, low_memory=False, encoding="utf-8", on_bad_lines="skip")

    df.replace("", pd.NA, inplace=True)
    return df


def save_checkpoint(path, user_sum, user_max, user_count, files_done):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "user_sum": dict(user_sum),
                "user_max": dict(user_max),
                "user_count": dict(user_count),
                "files_done": list(files_done),
            },
            f,
        )
    os.replace(tmp, path)
    print(f"  [CHECKPOINT] saved {len(files_done):,} files -> {path}", flush=True)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    start_time = time.time()
    text_fields = parse_text_fields(args.text_fields)
    files = sorted(glob.glob(args.csv_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.csv_glob}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_len
    emb_dim = model.get_sentence_embedding_dimension()

    use_cols = {"screen_name"} | {TEXT_FIELD_TO_COLUMN[field] for field in text_fields}

    print(f"Embedding model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Files: {len(files)}")
    print(f"Output: {args.out}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Resume: {args.resume}")
    print(f"Text fields: {','.join(text_fields)}")

    checkpoint = load_checkpoint(args.checkpoint_path) if args.resume else None
    if checkpoint:
        user_sum = defaultdict(lambda: np.zeros(emb_dim, dtype=np.float64), checkpoint.get("user_sum", {}))
        user_max = defaultdict(lambda: np.full(emb_dim, -np.inf, dtype=np.float32), checkpoint.get("user_max", {}))
        user_count = defaultdict(int, checkpoint.get("user_count", {}))
        files_done = set(checkpoint.get("files_done", []))
        print(
            f"Resumed from checkpoint: files_done={len(files_done):,} users={len(user_sum):,} "
            f"posts={sum(user_count.values()):,}",
            flush=True,
        )
    else:
        user_sum = defaultdict(lambda: np.zeros(emb_dim, dtype=np.float64))
        user_max = defaultdict(lambda: np.full(emb_dim, -np.inf, dtype=np.float32))
        user_count = defaultdict(int)
        files_done = set()

    processed_in_run = 0
    for i, fpath in enumerate(files, start=1):
        if fpath in files_done:
            print(f"[{i}/{len(files)}] Skipping {os.path.basename(fpath)} (already checkpointed)", flush=True)
            continue
        if args.max_files > 0 and processed_in_run >= args.max_files:
            print(f"Reached max_files={args.max_files}; stopping this run.", flush=True)
            break

        file_start = time.time()
        print(f"[{i}/{len(files)}] Loading {os.path.basename(fpath)}", flush=True)
        try:
            df = read_post_file(fpath)
        except Exception as exc:
            print(f"  [ERROR] Skipping {os.path.basename(fpath)}: {exc}", flush=True)
            continue

        if df.empty:
            print("  [SKIP] empty file", flush=True)
            continue

        cols = [c for c in df.columns if c in use_cols]
        if "screen_name" not in cols:
            print("  [SKIP] missing screen_name column", flush=True)
            continue
        df = df[cols].copy()
        raw_rows = len(df)

        df["screen_name"] = normalize_handle(df["screen_name"])
        df = df[df["screen_name"].notna()].copy()
        if df.empty:
            print(f"  [SKIP] no valid screen_name rows out of {raw_rows:,}", flush=True)
            continue

        texts = df.apply(build_text, axis=1, text_fields=text_fields).tolist()
        handles = df["screen_name"].tolist()

        valid_idx = [k for k, text in enumerate(texts) if text.strip()]
        if not valid_idx:
            print(f"  [SKIP] no non-empty texts out of {len(df):,} rows", flush=True)
            continue

        valid_texts = [texts[k] for k in valid_idx]
        valid_handles = [handles[k] for k in valid_idx]

        print(
            f"  rows={raw_rows:,} valid_handles={len(df):,} texts_to_embed={len(valid_texts):,}",
            flush=True,
        )

        embs = model.encode(
            valid_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for handle, emb in zip(valid_handles, embs):
            user_sum[handle] += emb.astype(np.float64)
            user_max[handle] = np.maximum(user_max[handle], emb.astype(np.float32))
            user_count[handle] += 1

        files_done.add(fpath)
        processed_in_run += 1
        elapsed = time.time() - file_start
        total_elapsed = time.time() - start_time
        print(
            "  [DONE] "
            f"file_time={elapsed:.1f}s total_time={total_elapsed/60:.1f}m "
            f"cumulative_users={len(user_sum):,} cumulative_posts={sum(user_count.values()):,}",
            flush=True,
        )
        if args.checkpoint_every > 0 and processed_in_run % args.checkpoint_every == 0:
            save_checkpoint(args.checkpoint_path, user_sum, user_max, user_count, files_done)

    save_checkpoint(args.checkpoint_path, user_sum, user_max, user_count, files_done)

    handles = sorted(user_sum.keys())
    n = len(handles)
    meanpool = torch.zeros(n, emb_dim, dtype=torch.float32)
    maxpool = torch.zeros(n, emb_dim, dtype=torch.float32)

    for idx, handle in enumerate(handles):
        meanpool[idx] = torch.from_numpy((user_sum[handle] / user_count[handle]).astype(np.float32))
        maxpool[idx] = torch.from_numpy(user_max[handle])

    out_obj = {
        "handles": handles,
        "meanpool": meanpool,
        "maxpool": maxpool,
        "counts": {handle: int(user_count[handle]) for handle in handles},
        "model": args.model,
    }
    torch.save(out_obj, args.out)

    meta = {
        "csv_glob": args.csv_glob,
        "files_count": len(files),
        "model": args.model,
        "text_fields": list(text_fields),
        "embedding_dim": int(emb_dim),
        "users": int(n),
        "total_posts_embedded": int(sum(user_count.values())),
    }
    with open(args.out.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings: {args.out}")
    print(f"users={n:,}, dim={emb_dim}")


if __name__ == "__main__":
    main()

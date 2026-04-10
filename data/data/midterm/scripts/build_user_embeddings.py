import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

TEXT_FIELD_TO_COLUMN = {
    "tweet_text": "text",
    "retweet_text": "rt_text",
    "bio": "description",
}
DEFAULT_TEXT_FIELDS = "tweet_text,retweet_text"


def parse_args():
    p = argparse.ArgumentParser(description="Build per-user (userid) pooled text embeddings from raw midterm CSV files.")
    p.add_argument("--csv_glob", default="/project2/ll_774_951/midterm/*/*.csv")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="data/data/midterm/embeddings/embeddings_all-MiniLM-L6-v2.pt")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument(
        "--text_fields",
        default=DEFAULT_TEXT_FIELDS,
        help="Comma-separated semantic text fields to embed. Supported: tweet_text,retweet_text,bio",
    )
    return p.parse_args()


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
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            value = str(v).strip()
            if value not in seen:
                parts.append(value)
                seen.add(value)
    if not parts:
        return ""
    return " ".join(parts)[:512]


def main():
    args = parse_args()
    text_fields = parse_text_fields(args.text_fields)
    files = sorted(glob.glob(args.csv_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.csv_glob}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_len
    emb_dim = model.get_sentence_embedding_dimension()

    use_cols = {"userid"} | {TEXT_FIELD_TO_COLUMN[field] for field in text_fields}

    user_sum = defaultdict(lambda: np.zeros(emb_dim, dtype=np.float64))
    user_max = defaultdict(lambda: np.full(emb_dim, -np.inf, dtype=np.float32))
    user_count = defaultdict(int)

    print(f"Embedding model: {args.model}")
    print(f"Files: {len(files)}")
    print(f"Text fields: {','.join(text_fields)}")

    for i, fpath in enumerate(files, start=1):
        df = pd.read_csv(fpath, low_memory=False, on_bad_lines="skip")
        if df.empty:
            continue

        cols = [c for c in df.columns if c in use_cols]
        if "userid" not in cols:
            continue
        df = df[cols].copy()

        df["userid"] = pd.to_numeric(df["userid"], errors="coerce")
        df = df.dropna(subset=["userid"])
        if df.empty:
            continue
        df["userid"] = df["userid"].astype(np.int64)

        texts = df.apply(build_text, axis=1, text_fields=text_fields).tolist()
        uids = df["userid"].tolist()

        valid_idx = [k for k, t in enumerate(texts) if t.strip()]
        if not valid_idx:
            continue

        valid_texts = [texts[k] for k in valid_idx]
        valid_uids = [uids[k] for k in valid_idx]

        embs = model.encode(
            valid_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        for uid, emb in zip(valid_uids, embs):
            user_sum[int(uid)] += emb.astype(np.float64)
            user_max[int(uid)] = np.maximum(user_max[int(uid)], emb.astype(np.float32))
            user_count[int(uid)] += 1

        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files")

    user_ids = np.array(sorted(user_sum.keys()), dtype=np.int64)
    n = len(user_ids)
    meanpool = torch.zeros(n, emb_dim, dtype=torch.float)
    maxpool = torch.zeros(n, emb_dim, dtype=torch.float)

    for idx, uid in enumerate(user_ids):
        meanpool[idx] = torch.from_numpy((user_sum[int(uid)] / user_count[int(uid)]).astype(np.float32))
        maxpool[idx] = torch.from_numpy(user_max[int(uid)])

    out_obj = {
        "user_ids": user_ids,
        "meanpool": meanpool,
        "maxpool": maxpool,
        "counts": {int(uid): int(user_count[int(uid)]) for uid in user_ids},
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

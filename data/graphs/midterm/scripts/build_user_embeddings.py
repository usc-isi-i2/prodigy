import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser(description="Build per-user (userid) pooled text embeddings from raw midterm CSV files.")
    p.add_argument("--csv_glob", default="/project2/ll_774_951/midterm/*/*.csv")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="data/graphs/midterm/embeddings/embeddings_all-MiniLM-L6-v2.pt")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max_seq_len", type=int, default=512)
    return p.parse_args()


# Midterm CSVs may contain interleaved 66-col + 11-col rows.
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
        next(reader, None)
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


def build_text(row: pd.Series) -> str:
    parts = []
    for col in ("text", "rt_text"):
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).strip())
            break
    bio = row.get("description")
    if pd.notna(bio) and str(bio).strip():
        parts.append(str(bio).strip())
    if not parts:
        return ""
    return " ".join(parts)[:512]


def main():
    args = parse_args()
    files = sorted(glob.glob(args.csv_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.csv_glob}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_len
    emb_dim = model.get_sentence_embedding_dimension()

    use_cols = {"userid", "text", "rt_text", "description"}

    user_sum = defaultdict(lambda: np.zeros(emb_dim, dtype=np.float64))
    user_max = defaultdict(lambda: np.full(emb_dim, -np.inf, dtype=np.float32))
    user_count = defaultdict(int)

    print(f"Embedding model: {args.model}")
    print(f"Files: {len(files)}")

    for i, fpath in enumerate(files, start=1):
        df = load_interleaved_csv(fpath)
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

        texts = df.apply(build_text, axis=1).tolist()
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

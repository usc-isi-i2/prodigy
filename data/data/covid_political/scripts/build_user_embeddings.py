"""
Encode covid_political user bios (profile column) with a sentence-transformer.

Each user has exactly one text (their bio), so there is no pooling across posts —
we just encode all profiles in one pass.

Output schema (same as other datasets so generate_graph.py can join by user_id):
    {
        "user_ids": np.ndarray[int64, (N,)],   # row indices from user_data (0..N-1)
        "handles":  list[None],                 # no screen names in this dataset
        "meanpool": torch.Tensor[float32, (N, D)],
        "counts":   np.ndarray[int64, (N,)],    # all ones (single bio per user)
        "model":    str,
    }
"""
import argparse
import json
import os
import shlex
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_CSV = "/scratch1/eibl/data/social_llm_covid/user_data.csv"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = "data/data/covid_political/embeddings/user_embeddings_minilm.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-user bio embeddings for the covid_political dataset."
    )
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max_seq_len", type=int, default=64)
    p.add_argument("--fp16", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    command = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])

    print(f"Loading user_data from {args.csv}")
    user_data = pd.read_csv(args.csv, index_col=0)
    print(f"Loaded {len(user_data):,} users")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_len
    if args.fp16 and args.device.startswith("cuda"):
        model = model.half()
    emb_dim = model.get_sentence_embedding_dimension()

    print(f"Model={args.model} device={args.device} fp16={args.fp16} "
          f"seq_len={args.max_seq_len} batch={args.batch_size}")

    # Fill missing bios with empty string so every user gets a vector.
    profiles = user_data["profile"].fillna("").tolist()
    n_empty = sum(1 for p in profiles if not str(p).strip())
    print(f"Empty/missing profiles: {n_empty:,} (will get zero-ish embeddings)")

    print(f"Encoding {len(profiles):,} profiles...", flush=True)
    embeddings = model.encode(
        profiles,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    user_ids = user_data.index.to_numpy(dtype=np.int64)
    counts = np.ones(len(user_ids), dtype=np.int64)

    out_obj = {
        "user_ids": user_ids,
        "handles": [None] * len(user_ids),
        "meanpool": torch.from_numpy(embeddings),
        "counts": counts,
        "model": args.model,
    }
    torch.save(out_obj, args.out)

    meta = {
        "csv": args.csv,
        "model": args.model,
        "embedding_dim": int(emb_dim),
        "users": int(len(user_ids)),
        "empty_profiles": int(n_empty),
        "command": command,
        "max_seq_len": args.max_seq_len,
        "fp16": bool(args.fp16),
        "wall_min": round((time.time() - t0) / 60, 2),
    }
    with open(args.out.replace(".pt", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {args.out}  users={len(user_ids):,} dim={emb_dim} "
          f"wall={(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()

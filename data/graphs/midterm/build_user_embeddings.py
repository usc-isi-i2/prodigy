"""
Build per-user MiniLM embeddings (meanpool + maxpool) from midterm Twitter CSV files.
Mirrors data/graphs/ukr_ru/instagram/build_user_embeddings.py.

Output: user_embeddings_minilm.pt
    handles   – list[str]           ordered screen_names
    meanpool  – Tensor[N, 384]
    maxpool   – Tensor[N, 384]
    counts    – dict[screen_name, int]
"""

import argparse
import gc
import glob
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_CSV_GLOB = "/project2/ll_774_951/midterm/*/*.csv"
DEFAULT_OUTPUT_PATH = "user_embeddings_minilm.pt"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_BATCH_SIZE = 256
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_USECOLS = "screen_name,text,rt_text,description"


def build_post_text(row: pd.Series) -> str:
    parts = []
    for col in ("text", "rt_text"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
            break
    bio = row.get("description")
    if pd.notna(bio) and str(bio).strip():
        parts.append(str(bio).strip())
    return " ".join(parts)[:512] if parts else ""


def save_checkpoint(checkpoint_path, user_sum, user_max, user_count, files_done):
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "user_sum":   dict(user_sum),
            "user_max":   dict(user_max),
            "user_count": dict(user_count),
            "files_done": files_done,
        }, f)
    os.replace(tmp, checkpoint_path)
    print(f"  Checkpoint saved ({len(files_done)} files done)")


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-user embeddings from Twitter CSV files."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_GLOB, help="CSV glob pattern")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH, help="Output .pt path")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint pickle path for resumable accumulation (default: derived from --out)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Save checkpoint every N processed files",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Max token length")
    parser.add_argument(
        "--usecols",
        default=DEFAULT_USECOLS,
        help="Comma-separated CSV columns to load when present",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of input files (0 means all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        out_dir = os.path.dirname(args.out)
        out_base = os.path.splitext(os.path.basename(args.out))[0]
        checkpoint_name = f"{out_base}_checkpoint.pkl"
        checkpoint_path = os.path.join(out_dir, checkpoint_name) if out_dir else checkpoint_name

    usecols = [c.strip() for c in args.usecols.split(",") if c.strip()]
    if "screen_name" not in usecols:
        usecols = ["screen_name"] + usecols

    print(f"Loading model {args.model} on {args.device}...")
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {dim}")

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        user_sum = defaultdict(lambda: np.zeros(dim, dtype=np.float64), checkpoint["user_sum"])
        user_max = defaultdict(lambda: np.full(dim, -np.inf, dtype=np.float32), checkpoint["user_max"])
        user_count = defaultdict(int, checkpoint["user_count"])
        files_done = set(checkpoint["files_done"])
        print(f"Resumed: {len(files_done)} files done, {len(user_sum)} users so far")
    else:
        user_sum = defaultdict(lambda: np.zeros(dim, dtype=np.float64))
        user_max = defaultdict(lambda: np.full(dim, -np.inf, dtype=np.float32))
        user_count = defaultdict(int)
        files_done = set()

    files = sorted(glob.glob(args.csv))
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f"Found {len(files)} CSV files ({len(files_done)} already done)")

    for fpath in tqdm(files, desc="Files"):
        if fpath in files_done:
            continue
        try:
            avail = pd.read_csv(fpath, nrows=0).columns.tolist()
            cols = [c for c in usecols if c in avail]
            df = pd.read_csv(fpath, usecols=cols, low_memory=False, on_bad_lines="skip")
        except Exception as e:
            print(f"  Skipping {os.path.basename(fpath)}: {e}")
            files_done.add(fpath)
            continue

        if "screen_name" not in df.columns:
            files_done.add(fpath)
            continue

        texts = df.apply(build_post_text, axis=1).tolist()
        handles = df["screen_name"].tolist()
        del df
        gc.collect()

        valid_idx = [i for i, (t, h) in enumerate(zip(texts, handles)) if t.strip() and pd.notna(h)]
        if not valid_idx:
            files_done.add(fpath)
            continue

        valid_texts = [texts[i] for i in valid_idx]
        valid_handles = [handles[i] for i in valid_idx]

        embeddings = model.encode(
            valid_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        for handle, emb in zip(valid_handles, embeddings):
            user_sum[handle] += emb.astype(np.float64)
            user_max[handle] = np.maximum(user_max[handle], emb)
            user_count[handle] += 1

        files_done.add(fpath)
        del valid_texts, valid_handles, embeddings
        gc.collect()
        if "cuda" in str(args.device).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.checkpoint_every > 0 and len(files_done) % args.checkpoint_every == 0:
            save_checkpoint(checkpoint_path, user_sum, user_max, user_count, files_done)

    handles_sorted = sorted(user_sum.keys())
    n_users = len(handles_sorted)
    print(f"\nTotal unique users: {n_users:,}")
    print(f"Total posts embedded: {sum(user_count.values()):,}")

    meanpool = torch.zeros(n_users, dim)
    maxpool = torch.zeros(n_users, dim)
    for i, h in enumerate(handles_sorted):
        meanpool[i] = torch.from_numpy((user_sum[h] / user_count[h]).astype(np.float32))
        maxpool[i] = torch.from_numpy(user_max[h])

    torch.save(
        {
            "handles": handles_sorted,
            "meanpool": meanpool,
            "maxpool": maxpool,
            "counts": {h: user_count[h] for h in handles_sorted},
        },
        args.out,
    )
    print(f"Saved to {args.out}  (meanpool: {meanpool.shape})")


if __name__ == "__main__":
    main()

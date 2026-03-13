"""
Build per-user MiniLM embeddings (meanpool + maxpool) from midterm Twitter CSV files.
Mirrors data/graphs/ukr_ru/instagram/build_user_embeddings.py.

Output: user_embeddings_minilm.pt
    handles   – list[str]           ordered screen_names
    meanpool  – Tensor[N, 384]
    maxpool   – Tensor[N, 384]
    counts    – dict[screen_name, int]
"""

import gc
import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_GLOB       = "/project2/ll_774_951/midterm/*/*.csv"
CHECKPOINT_PATH = "user_embeddings_minilm_checkpoint.pkl"
OUTPUT_PATH     = "user_embeddings_minilm.pt"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
CHECKPOINT_EVERY = 10
BATCH_SIZE      = 256
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

USECOLS = ["screen_name", "text", "description"]


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


def save_checkpoint(user_sum, user_max, user_count, files_done):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "user_sum":   dict(user_sum),
            "user_max":   dict(user_max),
            "user_count": dict(user_count),
            "files_done": files_done,
        }, f)
    os.replace(tmp, CHECKPOINT_PATH)
    print(f"  Checkpoint saved ({len(files_done)} files done)")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    with open(CHECKPOINT_PATH, "rb") as f:
        return pickle.load(f)


print(f"Loading model {MODEL_NAME} on {DEVICE}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.max_seq_length = 512
DIM = model.get_sentence_embedding_dimension()
print(f"Embedding dim: {DIM}")

checkpoint = load_checkpoint()
if checkpoint:
    user_sum   = defaultdict(lambda: np.zeros(DIM, dtype=np.float64), checkpoint["user_sum"])
    user_max   = defaultdict(lambda: np.full(DIM, -np.inf, dtype=np.float32), checkpoint["user_max"])
    user_count = defaultdict(int, checkpoint["user_count"])
    files_done = set(checkpoint["files_done"])
    print(f"Resumed: {len(files_done)} files done, {len(user_sum)} users so far")
else:
    user_sum   = defaultdict(lambda: np.zeros(DIM, dtype=np.float64))
    user_max   = defaultdict(lambda: np.full(DIM, -np.inf, dtype=np.float32))
    user_count = defaultdict(int)
    files_done = set()

files = sorted(glob.glob(DATA_GLOB))
print(f"Found {len(files)} CSV files ({len(files_done)} already done)")

for fpath in tqdm(files, desc="Files"):
    if fpath in files_done:
        continue
    try:
        avail = pd.read_csv(fpath, nrows=0).columns.tolist()
        usecols = [c for c in USECOLS if c in avail]
        df = pd.read_csv(fpath, usecols=usecols, low_memory=False, on_bad_lines="skip")
    except Exception as e:
        print(f"  Skipping {os.path.basename(fpath)}: {e}")
        files_done.add(fpath)
        continue

    if "screen_name" not in df.columns:
        files_done.add(fpath)
        continue

    texts   = df.apply(build_post_text, axis=1).tolist()
    handles = df["screen_name"].tolist()
    del df
    gc.collect()

    valid_idx = [i for i, (t, h) in enumerate(zip(texts, handles))
                 if t.strip() and pd.notna(h)]
    if not valid_idx:
        files_done.add(fpath)
        continue

    valid_texts   = [texts[i] for i in valid_idx]
    valid_handles = [handles[i] for i in valid_idx]

    embeddings = model.encode(
        valid_texts, batch_size=BATCH_SIZE,
        show_progress_bar=False, convert_to_numpy=True,
    )

    for handle, emb in zip(valid_handles, embeddings):
        user_sum[handle]   += emb.astype(np.float64)
        user_max[handle]    = np.maximum(user_max[handle], emb)
        user_count[handle] += 1

    files_done.add(fpath)
    del valid_texts, valid_handles, embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(files_done) % CHECKPOINT_EVERY == 0:
        save_checkpoint(user_sum, user_max, user_count, files_done)

handles_sorted = sorted(user_sum.keys())
N = len(handles_sorted)
print(f"\nTotal unique users: {N:,}")
print(f"Total posts embedded: {sum(user_count.values()):,}")

meanpool = torch.zeros(N, DIM)
maxpool  = torch.zeros(N, DIM)
for i, h in enumerate(handles_sorted):
    meanpool[i] = torch.from_numpy((user_sum[h] / user_count[h]).astype(np.float32))
    maxpool[i]  = torch.from_numpy(user_max[h])

torch.save({
    "handles":  handles_sorted,
    "meanpool": meanpool,
    "maxpool":  maxpool,
    "counts":   {h: user_count[h] for h in handles_sorted},
}, OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}  (meanpool: {meanpool.shape})")

"""
Build per-user embeddings (meanpool + maxpool) from Instagram post pickle files.
Processes files one-at-a-time to stay within memory limits.

Output: saves a dict to `user_embeddings.pt` with keys:
    - "handles":    list[str]           – ordered user handles
    - "meanpool":   Tensor[N, D]        – mean-pooled embedding per user
    - "maxpool":    Tensor[N, D]        – max-pooled embedding per user
    - "counts":     dict[str, int]      – post count per user
"""

import gc
import glob
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
DATA_GLOB       = "/project2/ll_774_951/uk_ru/Instagram_Uk_ru/*.pkl"
CHECKPOINT_PATH = "user_embeddings_minilm_checkpoint.pkl"
OUTPUT_PATH     = "user_embeddings_minilm.pt"                                           
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
CHECKPOINT_EVERY = 4  # save accumulator state every N files
BATCH_SIZE      = 256
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Helpers ─────────────────────────────────────────────────────────────────

def build_post_text(row: pd.Series) -> str:
    """Combine available text fields into a single string for embedding."""
    parts = []
    for col in ("description", "title", "caption", "message", "text"):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
            break
    if pd.notna(row.get("imageText")):
        parts.append(str(row["imageText"]).strip())
    return " ".join(parts) if parts else ""


def save_checkpoint(user_sum, user_max, user_count, files_done):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "user_sum":   dict(user_sum),
            "user_max":   dict(user_max),
            "user_count": dict(user_count),
            "files_done": files_done,
        }, f)
    os.replace(tmp, CHECKPOINT_PATH)  # atomic
    print(f"  ✓ Checkpoint saved ({len(files_done)} files done)")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    with open(CHECKPOINT_PATH, "rb") as f:
        return pickle.load(f)


# ── Load model ──────────────────────────────────────────────────────────────
print(f"Loading model {MODEL_NAME} on {DEVICE}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.max_seq_length = 512
DIM = model.get_sentence_embedding_dimension()
print(f"Embedding dim: {DIM}")

# ── Resume from checkpoint if available ─────────────────────────────────────
checkpoint = load_checkpoint()
if checkpoint:
    user_sum   = defaultdict(lambda: np.zeros(DIM, dtype=np.float64), checkpoint["user_sum"])
    user_max   = defaultdict(lambda: np.full(DIM, -np.inf, dtype=np.float32), checkpoint["user_max"])
    user_count = defaultdict(int, checkpoint["user_count"])
    files_done = set(checkpoint["files_done"])
    print(f"Resumed from checkpoint: {len(files_done)} files already processed, {len(user_sum)} users so far")
else:
    user_sum   = defaultdict(lambda: np.zeros(DIM, dtype=np.float64))
    user_max   = defaultdict(lambda: np.full(DIM, -np.inf, dtype=np.float32))
    user_count = defaultdict(int)
    files_done = set()

# ── Stream through files ────────────────────────────────────────────────────
files = sorted(glob.glob(DATA_GLOB))
print(f"Found {len(files)} files ({len(files_done)} already done)")

for i, fpath in enumerate(tqdm(files, desc="Files")):
    if fpath in files_done:
        continue

    try:
        df = pd.read_pickle(fpath)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
    except Exception as e:
        print(f"  ⚠ Skipping {os.path.basename(fpath)}: {e}")
        files_done.add(fpath)
        continue

    df['handle'] = df.account.str['handle']

    if "handle" not in df.columns:
        print(f"  ⚠ No 'handle' column in {os.path.basename(fpath)}, skipping")
        del df
        gc.collect()
        files_done.add(fpath)
        continue

    texts = df.apply(build_post_text, axis=1).tolist()
    handles = df["handle"].tolist()
    del df
    gc.collect()

    # Filter: must have text and a non-null handle
    valid_idx = [
        i for i, (t, h) in enumerate(zip(texts, handles))
        if t.strip() and pd.notna(h)
    ]
    if not valid_idx:
        files_done.add(fpath)
        continue

    valid_texts   = [texts[i] for i in valid_idx]
    valid_handles = [handles[i] for i in valid_idx]
    del texts, handles

    embeddings = model.encode(
        valid_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )  # shape (n, DIM), float32

    for handle, emb in zip(valid_handles, embeddings):
        user_sum[handle]   += emb.astype(np.float64)
        user_max[handle]    = np.maximum(user_max[handle], emb)
        user_count[handle] += 1

    files_done.add(fpath)
    del valid_texts, valid_handles, embeddings
    gc.collect()
    torch.cuda.empty_cache()

    if len(files_done) % CHECKPOINT_EVERY == 0:
        save_checkpoint(user_sum, user_max, user_count, files_done)

# ── Finalize ────────────────────────────────────────────────────────────────
handles_sorted = sorted(user_sum.keys())
N = len(handles_sorted)
print(f"\nTotal unique users: {N}")
print(f"Total posts embedded: {sum(user_count.values())}")

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

print(f"Saved to {OUTPUT_PATH}")
print(f"  meanpool shape: {meanpool.shape}")
print(f"  maxpool  shape: {maxpool.shape}")

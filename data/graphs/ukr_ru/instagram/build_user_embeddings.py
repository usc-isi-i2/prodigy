#!/usr/bin/env python3
"""
Build per-user embeddings (meanpool + maxpool) from Instagram post pickle files.
Processes files one-at-a-time to stay within memory limits.

Output: saves a dict to `user_embeddings.pt` with keys:
    - "handles":    list[str]           – ordered user handles
    - "meanpool":   Tensor[N, D]        – mean-pooled embedding per user
    - "maxpool":    Tensor[N, D]        – max-pooled embedding per user
    - "counts":     dict[str, int]      – post count per user
"""
import argparse
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

def parse_args():
    p = argparse.ArgumentParser(description="Build per-user embeddings from Instagram post pickles.")
    p.add_argument("--data-glob", default="/project2/ll_774_951/uk_ru/Instagram_Uk_ru/*.pkl",
                   help="glob pattern matching input .pkl files")
    p.add_argument("--checkpoint-path", default="user_embeddings_minilm_checkpoint.pkl",
                   help="path to checkpoint pickle")
    p.add_argument("--output-path", default="user_embeddings_minilm.pt",
                   help="path to final output .pt file")
    p.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence-transformers model name")
    p.add_argument("--checkpoint-every", type=int, default=4,
                   help="save accumulator state every N files")
    p.add_argument("--batch-size", type=int, default=256,
                   help="model.encode batch_size")
    p.add_argument("--device", default=None,
                   help='device to run on, e.g. "cpu" or "cuda". Default: auto-detect')
    p.add_argument("--max-files", type=int, default=None,
                   help="maximum number of new files to process (does not count files already done)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="start fresh even if checkpoint exists (default: resume if checkpoint exists)")
    p.set_defaults(resume=True)
    return p.parse_args()

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

def save_checkpoint(user_sum, user_max, user_count, files_done, checkpoint_path):
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "user_sum":   dict(user_sum),
            "user_max":   dict(user_max),
            "user_count": dict(user_count),
            "files_done": list(files_done),
        }, f)
    os.replace(tmp, checkpoint_path)  # atomic
    print(f"  ✓ Checkpoint saved ({len(files_done)} files done)")

def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # decide device
    if args.device:
        DEVICE = args.device
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_GLOB       = args.data_glob
    CHECKPOINT_PATH = args.checkpoint_path
    OUTPUT_PATH     = args.output_path
    MODEL_NAME      = args.model_name
    CHECKPOINT_EVERY = args.checkpoint_every
    BATCH_SIZE      = args.batch_size
    MAX_FILES       = args.max_files
    RESUME          = args.resume

    print("Config:")
    print(f"  data_glob       = {DATA_GLOB}")
    print(f"  checkpoint_path = {CHECKPOINT_PATH}")
    print(f"  output_path     = {OUTPUT_PATH}")
    print(f"  model_name      = {MODEL_NAME}")
    print(f"  checkpoint_every= {CHECKPOINT_EVERY}")
    print(f"  batch_size      = {BATCH_SIZE}")
    print(f"  device          = {DEVICE}")
    print(f"  max_files       = {MAX_FILES}")
    print(f"  resume          = {RESUME}")

    # ── Load model ──────────────────────────────────────────────────────────────
    print(f"Loading model {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    model.max_seq_length = 512
    DIM = model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {DIM}")

    # ── Resume from checkpoint if available ─────────────────────────────────────
    checkpoint = load_checkpoint(CHECKPOINT_PATH) if RESUME else None
    if checkpoint:
        # checkpoint stored lists/dicts - convert back to appropriate defaults
        user_sum   = defaultdict(lambda: np.zeros(DIM, dtype=np.float64), checkpoint.get("user_sum", {}))
        user_max   = defaultdict(lambda: np.full(DIM, -np.inf, dtype=np.float32), checkpoint.get("user_max", {}))
        user_count = defaultdict(int, checkpoint.get("user_count", {}))
        files_done = set(checkpoint.get("files_done", []))
        print(f"Resumed from checkpoint: {len(files_done)} files already processed, {len(user_sum)} users so far")
    else:
        user_sum   = defaultdict(lambda: np.zeros(DIM, dtype=np.float64))
        user_max   = defaultdict(lambda: np.full(DIM, -np.inf, dtype=np.float32))
        user_count = defaultdict(int)
        files_done = set()

    # ── Stream through files ────────────────────────────────────────────────────
    files = sorted(glob.glob(DATA_GLOB))
    print(f"Found {len(files)} files ({len(files_done)} already done)")

    processed_new_files = 0  # count only files we actually attempt now (not those in files_done)

    for fpath in tqdm(files, desc="Files"):
        if fpath in files_done:
            continue

        # check if we've reached user's max_files limit
        if MAX_FILES is not None and processed_new_files >= MAX_FILES:
            print(f"Reached --max-files={MAX_FILES}; stopping early.")
            break

        processed_new_files += 1

        try:
            df = pd.read_pickle(fpath)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        except Exception as e:
            print(f"  ⚠ Skipping {os.path.basename(fpath)}: {e}")
            files_done.add(fpath)
            continue

        # some pickles may have nested structures; expect 'account' dict-like with 'handle'
        if "account" in df.columns and isinstance(df.account.iloc[0], (dict,)) and 'handle' not in df.columns:
            try:
                df['handle'] = df.account.str['handle']
            except Exception:
                # proceed; build_post_text / handle existence check below will catch problems
                pass

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
        # free cuda cache if available
        if DEVICE.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if len(files_done) % CHECKPOINT_EVERY == 0:
            save_checkpoint(user_sum, user_max, user_count, files_done, CHECKPOINT_PATH)

    # final checkpoint save (if anything new happened)
    save_checkpoint(user_sum, user_max, user_count, files_done, CHECKPOINT_PATH)

    # ── Finalize ───────────────────────────────────────────────────────────────
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

if __name__ == "__main__":
    main()
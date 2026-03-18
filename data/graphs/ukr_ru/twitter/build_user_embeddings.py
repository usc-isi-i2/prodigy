"""
Build per-user embeddings (meanpool + maxpool) from Instagram/Twitter post CSV files.
Processes files one-at-a-time to stay within memory limits.

This version reads CSVs (including the interleaved-format CSVs) instead of pickles.
"""
import gc
import glob
import os
import pickle
import csv
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
DATA_GLOB       = "/project2/ll_774_951/uk_ru/twitter/data/2022-02/*.csv"
CHECKPOINT_PATH = "user_embeddings_minilm_checkpoint.pkl"
OUTPUT_PATH     = "user_embeddings_minilm.pt"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
CHECKPOINT_EVERY = 4  # save accumulator state every N files
BATCH_SIZE      = 256
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Helpers: CSV loader for interleaved files ─────────────────────────────────

def load_interleaved_csv(filepath):
    """
    Read CSV files with interleaved main-row (len 66) + optional sub-row (len 11).
    If the file isn't in this format, this function may raise or return a DataFrame with mismatched columns.
    """
    main_rows, sub_rows = [], []

    # First read the header and sub-header to get column names
    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("Empty CSV")
        try:
            sub_header_raw = next(reader)
        except StopIteration:
            raise ValueError("CSV missing expected second (sub) header line")

    # Now iterate and pair main rows with sub rows. Handles trailing main without sub.
    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        next(reader)  # skip sub-header

        pending_main = None
        for row in reader:
            # skip empty rows
            if not any(cell.strip() for cell in row):
                continue
            if len(row) == len(header):
                # a main row
                if pending_main is not None:
                    # no sub row for previous main: push empty sub
                    main_rows.append(pending_main)
                    sub_rows.append([""] * len(sub_header_raw))
                pending_main = row
            elif len(row) == len(sub_header_raw):
                # a sub row
                if pending_main is not None:
                    main_rows.append(pending_main)
                    sub_rows.append(row)
                    pending_main = None
                else:
                    # a sub row without a preceding main row: skip
                    continue
            else:
                # unknown-length row: try to be lenient: if pending_main present and row shorter, treat as sub (pad)
                if pending_main is not None and len(row) <= len(sub_header_raw):
                    padded = row + [""] * (len(sub_header_raw) - len(row))
                    main_rows.append(pending_main)
                    sub_rows.append(padded)
                    pending_main = None
                else:
                    # if it's unexpected, skip
                    continue

        if pending_main is not None:
            main_rows.append(pending_main)
            sub_rows.append([""] * len(sub_header_raw))

    # Build dataframes and concat side-by-side (drop the first sub column if it's an extra)
    # The original sub_cols in your example had an extra leading "sub_extra" to drop.
    # Here we infer sub column names from the sub_header_raw; if there's an extra leading column,
    # we'll drop it if its name is blank or obviously a placeholder.
    sub_cols = list(sub_header_raw)
    # If first sub column is empty-ish, rename to 'sub_extra' and drop later
    if not sub_cols or not sub_cols[0].strip():
        sub_cols[0] = "sub_extra"

    df_main = pd.DataFrame(main_rows, columns=header)
    df_sub  = pd.DataFrame(sub_rows, columns=sub_cols)

    # If 'sub_extra' exists, drop it to match previous expected layout
    if "sub_extra" in df_sub.columns:
        df_sub = df_sub.drop(columns=["sub_extra"])

    df = pd.concat([df_main.reset_index(drop=True), df_sub.reset_index(drop=True)], axis=1)
    return df

def try_parse_json_field(val):
    """If val is a JSON string for a dict-like field, try to parse it."""
    if not isinstance(val, str):
        return val
    val = val.strip()
    if (val.startswith("{") and val.endswith("}")) or (val.startswith('"{"') and val.endswith('}"')):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val

def read_post_file(fpath):
    """
    Read a single file robustly:
      - attempt interleaved parser first (load_interleaved_csv)
      - fall back to pd.read_csv with low_memory=False
    Also ensures 'handle' column exists if possible by checking a few common places:
      - 'handle' column
      - 'account.handle' column
      - 'account' column containing JSON/dict with 'handle' key
    Returns a DataFrame or raises an exception.
    """
    # Try interleaved loader first (because some files are interleaved in your dataset)
    try:
        df = load_interleaved_csv(fpath)
        # success; convert empty strings to NaN consistently
        df.replace("", pd.NA, inplace=True)
    except Exception as e_inter:
        # Fall back to standard CSV read
        try:
            df = pd.read_csv(fpath, low_memory=False, encoding="utf-8", error_bad_lines=False)
            df.replace("", pd.NA, inplace=True)
        except Exception as e_csv:
            raise RuntimeError(f"Failed to read {os.path.basename(fpath)} as interleaved CSV ({e_inter}) or regular CSV ({e_csv})")

    # Ensure handle column: several formats possible
    if "handle" not in df.columns:
        # pattern: account.handle
        if "account.handle" in df.columns:
            df["handle"] = df["account.handle"]
        elif "account" in df.columns:
            # account may be a JSON string or a dict-like object; try to parse/extract
            try:
                # attempt to parse JSON-like strings when present
                parsed = df["account"].apply(try_parse_json_field)
                if parsed.apply(lambda x: isinstance(x, dict)).any():
                    df["handle"] = parsed.apply(lambda a: a.get("handle") if isinstance(a, dict) else pd.NA)
                else:
                    # maybe account column is like "handle:foo" or has a handle substring; skip heuristic
                    # leave for downstream; no handle column created here
                    pass
            except Exception:
                pass

    return df

# ── Helpers for text building and checkpointing ──────────────────────────────

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
        df = read_post_file(fpath)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
    except Exception as e:
        print(f"  ⚠ Skipping {os.path.basename(fpath)}: read error: {e}")
        files_done.add(fpath)
        continue

    # Ensure we can find a handle column; try a couple of heuristics
    if "handle" not in df.columns:
        # try other common names
        possible = None
        for c in ("screen_name", "username", "user_screen_name", "user"):
            if c in df.columns:
                possible = c
                break
        if possible is not None:
            df["handle"] = df[possible]
        else:
            # nothing we can use reliably
            print(f"  ⚠ No 'handle' column (or equivalent) in {os.path.basename(fpath)}, skipping")
            del df
            gc.collect()
            files_done.add(fpath)
            continue

    # If account column exists and is dict-like with handle, prefer that for reliability
    if "account" in df.columns and "handle" not in df.columns:
        try:
            parsed = df["account"].apply(try_parse_json_field)
            if parsed.apply(lambda x: isinstance(x, dict) and "handle" in x).any():
                df["handle"] = parsed.apply(lambda a: a.get("handle") if isinstance(a, dict) else pd.NA)
        except Exception:
            pass

    # Build texts and handles
    texts = df.apply(build_post_text, axis=1).tolist()
    handles = df["handle"].tolist()
    del df
    gc.collect()

    # Filter: must have text and a non-null handle
    valid_idx = [
        idx for idx, (t, h) in enumerate(zip(texts, handles))
        if isinstance(t, str) and t.strip() and pd.notna(h)
    ]
    if not valid_idx:
        files_done.add(fpath)
        continue

    valid_texts   = [texts[idx] for idx in valid_idx]
    valid_handles = [handles[idx] for idx in valid_idx]
    del texts, handles

    embeddings = model.encode(
        valid_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )  # shape (n, DIM), float32

    for handle, emb in zip(valid_handles, embeddings):
        # ensure handle is string (strip whitespace)
        handle = str(handle).strip()
        user_sum[handle]   += emb.astype(np.float64)
        user_max[handle]    = np.maximum(user_max[handle], emb)
        user_count[handle] += 1

    files_done.add(fpath)
    del valid_texts, valid_handles, embeddings
    gc.collect()
    if DEVICE.startswith("cuda"):
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
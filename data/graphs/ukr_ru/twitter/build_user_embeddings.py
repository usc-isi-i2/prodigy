#!/usr/bin/env python3
"""
Build per-user embeddings (meanpool + maxpool) from Instagram/Twitter post CSV files.
Processes files one-at-a-time to stay within memory limits.

Usage example:
  python build_user_embeddings.py --data-glob "/path/*.csv" --output-path out.pt --max-files 100
"""
import argparse
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

# -------------------------
# Helpers: CSV loader for interleaved files
# -------------------------

def load_interleaved_csv(filepath):
    """
    Read CSV files with interleaved main-row (len header) + optional sub-row (len sub_header).
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

    sub_cols = list(sub_header_raw)
    if not sub_cols or not sub_cols[0].strip():
        sub_cols[0] = "sub_extra"

    df_main = pd.DataFrame(main_rows, columns=header)
    df_sub  = pd.DataFrame(sub_rows, columns=sub_cols)

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
    Also ensures 'handle' column exists if possible by checking a few common places.
    Returns a DataFrame or raises an exception.
    """
    # Try interleaved loader first (because some files are interleaved in some datasets)
    try:
        df = load_interleaved_csv(fpath)
        df.replace("", pd.NA, inplace=True)
    except Exception as e_inter:
        # Fall back to standard CSV read
        try:
            # pandas' error_bad_lines is deprecated; use on_bad_lines
            df = pd.read_csv(fpath, low_memory=False, encoding="utf-8", on_bad_lines="skip")
            df.replace("", pd.NA, inplace=True)
        except Exception as e_csv:
            raise RuntimeError(f"Failed to read {os.path.basename(fpath)} as interleaved CSV ({e_inter}) or regular CSV ({e_csv})")

    # Ensure handle column: several formats possible
    if "handle" not in df.columns:
        if "account.handle" in df.columns:
            df["handle"] = df["account.handle"]
        elif "account" in df.columns:
            try:
                parsed = df["account"].apply(try_parse_json_field)
                if parsed.apply(lambda x: isinstance(x, dict)).any():
                    df["handle"] = parsed.apply(lambda a: a.get("handle") if isinstance(a, dict) else pd.NA)
                else:
                    pass
            except Exception:
                pass

    return df

# -------------------------
# Helpers for text building and checkpointing
# -------------------------

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

def save_checkpoint(path, user_sum, user_max, user_count, files_done):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({
            "user_sum":   dict(user_sum),
            "user_max":   dict(user_max),
            "user_count": dict(user_count),
            "files_done": list(files_done),
        }, f)
    os.replace(tmp, path)  # atomic
    print(f"  ✓ Checkpoint saved ({len(files_done)} files done) -> {path}")

def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser(description="Build per-user embeddings (mean + max pooling) from post CSVs.")
    p.add_argument("--data-glob", type=str, required=True, help="glob pattern for input CSV files (e.g. '/data/*.csv')")
    p.add_argument("--checkpoint-path", type=str, default="user_embeddings_minilm_checkpoint.pkl", help="checkpoint pickle path")
    p.add_argument("--output-path", type=str, default="user_embeddings_minilm.pt", help="output torch .pt file")
    p.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    p.add_argument("--checkpoint-every", type=int, default=4, help="save accumulator state every N files")
    p.add_argument("--batch-size", type=int, default=256, help="encoding batch size")
    p.add_argument("--device", type=str, default=None, help="device to use (e.g. 'cuda' or 'cpu'); defaults to auto-detect")
    p.add_argument("--max-files", type=int, default=None, help="maximum number of files to process in this run (unprocessed files only). None => all files")
    p.add_argument("--resume", action="store_true", default=False, help="resume from checkpoint if available")
    p.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA even if available")
    p.add_argument("--verbose", action="store_true", default=False, help="print extra logs")
    args = p.parse_args()

    # Resolve device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Print config
    print("Configuration:")
    for k in ("data_glob", "checkpoint_path", "output_path", "model_name", "checkpoint_every", "batch_size", "device", "max_files", "resume"):
        print(f"  {k}: {getattr(args, k)}")
    print("")

    # Load model
    print(f"Loading model {args.model_name} on {device}...")
    model = SentenceTransformer(args.model_name, device=device)
    model.max_seq_length = 512
    DIM = model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {DIM}")

    # Resume or init accumulators
    checkpoint = load_checkpoint(args.checkpoint_path) if args.resume else None
    if checkpoint:
        # loaded checkpoint stores lists/dicts -> need to wrap back into default dicts
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

    # Discover files
    files = sorted(glob.glob(args.data_glob))
    print(f"Found {len(files)} files (already done: {len(files_done)})")

    processed_in_run = 0
    for i, fpath in enumerate(tqdm(files, desc="Files")):
        # if file already processed (from checkpoint), skip
        if fpath in files_done:
            if args.verbose:
                print(f"Skipping already-processed file: {fpath}")
            continue

        # apply max-files limit (only counts newly processed files)
        if args.max_files is not None and processed_in_run >= args.max_files:
            if args.verbose:
                print(f"Reached max-files={args.max_files}; stopping.")
            break

        try:
            df = read_post_file(fpath)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        except Exception as e:
            print(f"  ⚠ Skipping {os.path.basename(fpath)}: read error: {e}")
            files_done.add(fpath)
            processed_in_run += 1  # consider it consumed for the max-files counter
            continue

        # ensure we can find a handle column; try a couple heuristics
        if "handle" not in df.columns:
            possible = None
            for c in ("screen_name", "username", "user_screen_name", "user"):
                if c in df.columns:
                    possible = c
                    break
            if possible is not None:
                df["handle"] = df[possible]
            else:
                print(f"  ⚠ No 'handle' column (or equivalent) in {os.path.basename(fpath)}, skipping")
                del df
                gc.collect()
                files_done.add(fpath)
                processed_in_run += 1
                continue

        # if account column exists and is dict-like with handle, prefer that
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
            processed_in_run += 1
            continue

        valid_texts   = [texts[idx] for idx in valid_idx]
        valid_handles = [handles[idx] for idx in valid_idx]
        del texts, handles

        # encode embeddings
        embeddings = model.encode(
            valid_texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )  # shape (n, DIM), float32

        for handle, emb in zip(valid_handles, embeddings):
            handle = str(handle).strip()
            user_sum[handle]   += emb.astype(np.float64)
            user_max[handle]    = np.maximum(user_max[handle], emb)
            user_count[handle] += 1

        files_done.add(fpath)
        processed_in_run += 1

        # free memory
        del valid_texts, valid_handles, embeddings
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        if len(files_done) % args.checkpoint_every == 0:
            save_checkpoint(args.checkpoint_path, user_sum, user_max, user_count, files_done)

    # Final checkpoint before finalizing (safe to always save)
    save_checkpoint(args.checkpoint_path, user_sum, user_max, user_count, files_done)

    # Finalize
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
    }, args.output_path)

    print(f"Saved to {args.output_path}")
    print(f"  meanpool shape: {meanpool.shape}")
    print(f"  maxpool  shape: {maxpool.shape}")

if __name__ == "__main__":
    main()
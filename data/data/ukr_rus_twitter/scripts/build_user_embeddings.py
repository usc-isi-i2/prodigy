"""Build per-user mean-pooled text embeddings from ukr_rus_twitter CSV files.

Uses screen_name as the identity key (numeric user ids are best-effort only).
"""
import argparse
import glob
import json
import os
import shlex
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from rapids.embeddings.pipeline import finalize_embeddings, run_embedding_pipeline
from rapids.loaders.csv_loader import load_ukr_rus_file
from rapids.utils import normalize_handle

DEFAULT_CSV = "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = "data/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-user pooled text embeddings from ukr_rus_twitter CSV files."
    )
    p.add_argument("--csv_glob", default=DEFAULT_CSV)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=0)
    p.add_argument(
        "--stop_after_max_nodes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max_seq_len", type=int, default=64)
    p.add_argument("--fp16", action="store_true", default=True)
    return p.parse_args()


def _get_records(fpath: str):
    df = load_ukr_rus_file(fpath)
    if df.empty or "screen_name" not in df.columns:
        return []
    df["screen_name"] = df["screen_name"].apply(normalize_handle)
    df = df[df["screen_name"].notna()].copy()
    if "userid" in df.columns:
        df["userid"] = pd.to_numeric(df["userid"], errors="coerce")
    return df.to_dict("records")


def _get_uid(record: dict):
    return normalize_handle(record.get("screen_name"))


def _get_text(record: dict) -> str:
    rt = record.get("rt_text")
    if rt and str(rt).strip() and str(rt) not in {"nan", "<NA>"}:
        return str(rt).strip()
    own = record.get("text")
    if own and str(own).strip() and str(own) not in {"nan", "<NA>"}:
        return str(own).strip()
    return ""


def main():
    args = parse_args()
    command = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])

    files = sorted(glob.glob(args.csv_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.csv_glob}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_len
    if args.fp16 and args.device.startswith("cuda"):
        model = model.half()
    emb_dim = model.get_sentence_embedding_dimension()

    print(f"Model={args.model} device={args.device} fp16={args.fp16} "
          f"seq_len={args.max_seq_len} batch={args.batch_size} files={len(files)}")

    # Collect best-effort userid per handle as we process files
    _handle_to_userid: dict = {}

    _orig_get_records = _get_records

    def _get_records_tracking(fpath: str):
        records = _orig_get_records(fpath)
        for r in records:
            handle = normalize_handle(r.get("screen_name"))
            uid = r.get("userid")
            if handle and uid is not None and not pd.isna(uid) and handle not in _handle_to_userid:
                try:
                    _handle_to_userid[handle] = int(uid)
                except Exception:
                    pass
        return records

    uid_to_row, sum_mat, cnt_arr, stats = run_embedding_pipeline(
        files=files,
        model=model,
        get_records=_get_records_tracking,
        get_uid=_get_uid,
        get_text=_get_text,
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        stop_after_max_nodes=args.stop_after_max_nodes,
    )

    # keys are screen_names (handles)
    handles, meanpool, counts = finalize_embeddings(uid_to_row, sum_mat, cnt_arr, args.max_nodes)
    user_ids = [_handle_to_userid.get(h) for h in handles]

    torch.save({
        "handles": handles,
        "user_ids": user_ids,
        "meanpool": meanpool,
        "counts": counts,
        "model": args.model,
    }, args.out)

    meta = {
        "csv_glob": args.csv_glob,
        "files_count": len(files),
        "model": args.model,
        "embedding_dim": int(emb_dim),
        "users": int(len(handles)),
        "max_nodes": int(args.max_nodes),
        "stop_after_max_nodes": bool(args.stop_after_max_nodes),
        "command": command,
        "max_seq_len": args.max_seq_len,
        "fp16": bool(args.fp16),
        **stats,
    }
    with open(args.out.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {args.out} users={len(handles):,} dim={emb_dim} "
          f"posts={stats['total_posts_embedded']:,} wall={stats['elapsed_min']:.1f}m")


if __name__ == "__main__":
    main()

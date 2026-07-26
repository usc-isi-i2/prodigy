"""Embed CP-HK user bios from ``user_bios.parquet``.

This is similar to ``scripts/social_llm/build_bio_embeddings.py`` but supports
hashed string user ids and the CP-HK parquet staging format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bio_embeddings.constants import PREPROCESSING_VERSION
from scripts.bio_embeddings.preprocessing import bio_hash, normalize_bio_text
from scripts.tweet_embeddings.constants import DEFAULT_MODEL, DEFAULT_REVISION, OUTPUT_DIM
from scripts.tweet_embeddings.model_backend import compute_token_lengths, encode_texts, load_model_for_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users-parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bio-output-root", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=OUTPUT_DIM)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-folder", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    command = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])

    users = pd.read_parquet(args.users_parquet)
    required = {"node_id", "user_id", "profile"}
    missing = required - set(users.columns)
    if missing:
        raise KeyError(f"{args.users_parquet} missing columns: {sorted(missing)}")
    users = users.sort_values("node_id").reset_index(drop=True)
    if not (users["node_id"].to_numpy() == np.arange(len(users))).all():
        raise ValueError("node_id must be contiguous and 0-indexed in users parquet.")

    normalized = users["profile"].fillna("").map(normalize_bio_text).tolist()
    unique_by_hash: dict[str, str] = {}
    bio_hashes: list[str] = []
    for text in normalized:
        digest = bio_hash(text) if text else ""
        bio_hashes.append(digest)
        if digest and digest not in unique_by_hash:
            unique_by_hash[digest] = text
    ordered_hashes = sorted(unique_by_hash)
    ordered_texts = [unique_by_hash[digest] for digest in ordered_hashes]
    row_for_hash = {digest: idx for idx, digest in enumerate(ordered_hashes)}
    row_to_unique = np.asarray([row_for_hash[digest] if digest else -1 for digest in bio_hashes], dtype=np.int64)

    cfg = {
        "model": args.model,
        "revision": args.revision,
        "embedding_dim": int(args.embedding_dim),
        "batch_size": int(args.batch_size),
        "max_seq_length": int(args.max_seq_length),
        "fp16": bool(args.fp16),
        "cpu": bool(args.cpu),
        "cache_folder": args.cache_folder or "",
    }
    print(
        f"Encoding CP-HK bios: users={len(users):,} unique_nonempty={len(ordered_texts):,} "
        f"model={args.model} revision={args.revision}",
        flush=True,
    )
    model, device = load_model_for_worker(0, cfg)
    unique_embeddings = encode_texts(model, ordered_texts, cfg)
    if unique_embeddings.shape != (len(ordered_texts), int(args.embedding_dim)):
        raise RuntimeError(f"Unexpected embedding shape: {unique_embeddings.shape}")

    embeddings = np.zeros((len(users), int(args.embedding_dim)), dtype=np.float16)
    has_bio = row_to_unique >= 0
    if has_bio.any():
        embeddings[has_bio] = unique_embeddings[row_to_unique[has_bio]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_obj = {
        "node_ids": users["node_id"].to_numpy(dtype=np.int64),
        "user_ids": users["user_id"].astype(str).tolist(),
        "meanpool": torch.from_numpy(embeddings),
        "counts": has_bio.astype(np.int64),
        "bio_hashes": bio_hashes,
        "model": args.model,
        "revision": args.revision,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    torch.save(out_obj, out_path)

    token_lengths, truncation_flags = compute_token_lengths(model, ordered_texts, int(args.max_seq_length))
    meta = {
        "users_parquet": args.users_parquet,
        "out": str(out_path),
        "users": int(len(users)),
        "unique_nonempty_bios": int(len(ordered_texts)),
        "empty_bios": int((~has_bio).sum()),
        "model": args.model,
        "revision": args.revision,
        "embedding_dim": int(args.embedding_dim),
        "embedding_dtype": "float16",
        "preprocessing_version": PREPROCESSING_VERSION,
        "max_seq_length": int(args.max_seq_length),
        "batch_size": int(args.batch_size),
        "device": device,
        "command": command,
        "wall_min": round((time.time() - started) / 60, 2),
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    if args.bio_output_root:
        bio_root = Path(args.bio_output_root)
        bio_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "bio_id": np.arange(len(ordered_hashes), dtype=np.int64),
                "bio_hash": ordered_hashes,
                "normalized_bio_text": ordered_texts,
                "token_length": token_lengths,
                "truncation_flag": truncation_flags,
                "model": args.model,
                "revision": args.revision,
                "preprocessing_version": PREPROCESSING_VERSION,
            }
        ).to_csv(bio_root / "bio_texts.csv", index=False)
        with (bio_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
    print(f"Saved CP-HK bio embeddings: {out_path}", flush=True)


if __name__ == "__main__":
    main()

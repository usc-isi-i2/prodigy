"""
Build row-aligned social_llm profile-bio embeddings with the GTE bio pipeline model.

The output .pt is intentionally compatible with scripts/social_llm/generate_graph.py:

    {
        "user_ids": np.ndarray[int64, (N,)],
        "handles": list[None],
        "meanpool": torch.Tensor[float16, (N, 768)],
        "counts": np.ndarray[int64, (N,)],
        ...
    }

Missing/empty bios receive an all-zero vector and counts=0.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from scripts.bio_embeddings.constants import PREPROCESSING_VERSION
from scripts.bio_embeddings.preprocessing import bio_hash, normalize_bio_text
from scripts.tweet_embeddings.constants import DEFAULT_MODEL, DEFAULT_REVISION, OUTPUT_DIM
from scripts.tweet_embeddings.model_backend import (
    compute_token_lengths,
    encode_texts,
    load_model_for_worker,
)


DEFAULT_OUT = "scripts/social_llm/embeddings/user_bio_embeddings_gte_multilingual_base.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed social_llm user_data.csv profile bios with Alibaba-NLP/gte-multilingual-base."
    )
    parser.add_argument("--csv", required=True, help="Path to user_data.csv.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output row-aligned .pt file.")
    parser.add_argument(
        "--bio-output-root",
        default="",
        help="Optional directory for bio_texts.csv and embedding metadata sidecars.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=OUTPUT_DIM)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-folder", default="")
    return parser.parse_args()


def read_user_data_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (UnicodeDecodeError, pd.errors.ParserError):
        raw = Path(path).read_bytes().replace(b"\x00", b"")
        text = raw.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(text), engine="python")


def build_unique_bios(profiles: pd.Series):
    normalized = profiles.fillna("").map(normalize_bio_text).tolist()
    bio_hashes = [bio_hash(text) if text else "" for text in normalized]

    unique_by_hash: dict[str, str] = {}
    for digest, text in zip(bio_hashes, normalized):
        if digest and digest not in unique_by_hash:
            unique_by_hash[digest] = text

    ordered_hashes = sorted(unique_by_hash)
    ordered_texts = [unique_by_hash[digest] for digest in ordered_hashes]
    row_to_unique = {digest: i for i, digest in enumerate(ordered_hashes)}
    user_unique_rows = np.array(
        [row_to_unique[digest] if digest else -1 for digest in bio_hashes],
        dtype=np.int64,
    )
    return normalized, bio_hashes, ordered_hashes, ordered_texts, user_unique_rows


def main():
    args = parse_args()
    t0 = time.time()
    command = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])

    print(f"Loading user_data: {args.csv}")
    user_data = read_user_data_csv(args.csv)
    if "profile" not in user_data.columns:
        raise KeyError(f"{args.csv} must contain a 'profile' column.")
    print(f"Loaded {len(user_data):,} rows")

    normalized, bio_hashes, unique_hashes, unique_texts, user_unique_rows = build_unique_bios(
        user_data["profile"]
    )
    empty_bios = int((user_unique_rows < 0).sum())
    print(
        f"Profiles: users={len(user_data):,} unique_nonempty={len(unique_texts):,} "
        f"empty={empty_bios:,}"
    )

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
    model, device = load_model_for_worker(0, cfg)
    print(
        f"Encoding with model={args.model} revision={args.revision} "
        f"device={device} batch={args.batch_size} max_seq_length={args.max_seq_length}"
    )

    unique_embeddings = encode_texts(model, unique_texts, cfg)
    if unique_embeddings.shape != (len(unique_texts), int(args.embedding_dim)):
        raise RuntimeError(f"Unexpected embedding shape: {unique_embeddings.shape}")

    user_embeddings = np.zeros((len(user_data), int(args.embedding_dim)), dtype=np.float16)
    has_bio = user_unique_rows >= 0
    if has_bio.any():
        user_embeddings[has_bio] = unique_embeddings[user_unique_rows[has_bio]]

    token_lengths, truncation_flags = compute_token_lengths(
        model, unique_texts, int(args.max_seq_length)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    user_ids = user_data.index.to_numpy(dtype=np.int64)
    counts = has_bio.astype(np.int64)
    out_obj = {
        "user_ids": user_ids,
        "handles": [None] * len(user_ids),
        "meanpool": torch.from_numpy(user_embeddings),
        "counts": counts,
        "bio_hashes": bio_hashes,
        "model": args.model,
        "revision": args.revision,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    torch.save(out_obj, out_path)

    meta = {
        "csv": args.csv,
        "out": str(out_path),
        "users": int(len(user_ids)),
        "unique_nonempty_bios": int(len(unique_texts)),
        "empty_bios": empty_bios,
        "model": args.model,
        "revision": args.revision,
        "embedding_dim": int(args.embedding_dim),
        "embedding_dtype": "float16",
        "preprocessing_version": PREPROCESSING_VERSION,
        "max_seq_length": int(args.max_seq_length),
        "batch_size": int(args.batch_size),
        "fp16": bool(args.fp16),
        "device": device,
        "command": command,
        "wall_min": round((time.time() - t0) / 60, 2),
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    if args.bio_output_root:
        bio_root = Path(args.bio_output_root)
        bio_root.mkdir(parents=True, exist_ok=True)
        bio_rows = pd.DataFrame(
            {
                "bio_id": np.arange(len(unique_hashes), dtype=np.int64),
                "bio_hash": unique_hashes,
                "normalized_bio_text": unique_texts,
                "token_length": token_lengths,
                "truncation_flag": truncation_flags,
                "model": args.model,
                "revision": args.revision,
                "preprocessing_version": PREPROCESSING_VERSION,
            }
        )
        bio_rows.to_csv(bio_root / "bio_texts.csv", index=False)
        with (bio_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    print(f"Saved embeddings: {out_path}")
    print(f"Saved meta: {out_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()

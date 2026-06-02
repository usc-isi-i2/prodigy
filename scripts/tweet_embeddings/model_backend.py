"""SentenceTransformers model loading and batched encoding."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def compute_token_lengths(model: Any, texts: list[str], max_seq_length: int) -> tuple[list[int], list[bool]]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return [-1] * len(texts), [False] * len(texts)
    lengths: list[int] = []
    flags: list[bool] = []
    step = 2048
    for start in range(0, len(texts), step):
        batch = texts[start : start + step]
        try:
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_length=True,
            )
            batch_lengths = encoded.get("length")
            if batch_lengths is None:
                batch_lengths = [len(ids) for ids in encoded["input_ids"]]
        except Exception:
            batch_lengths = [-1] * len(batch)
        lengths.extend(int(value) for value in batch_lengths)
        flags.extend(bool(value > max_seq_length) if value >= 0 else False for value in batch_lengths)
    return lengths, flags


def encode_texts(model: Any, texts: list[str], cfg: dict[str, Any]) -> np.ndarray:
    if not texts:
        return np.empty((0, int(cfg["embedding_dim"])), dtype=np.float16)
    chunks: list[np.ndarray] = []
    encode_batch_size = int(cfg["batch_size"])
    for start in range(0, len(texts), encode_batch_size):
        batch = texts[start : start + encode_batch_size]
        embeddings = model.encode(
            batch,
            batch_size=encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        chunks.append(np.asarray(embeddings, dtype=np.float16))
    return np.vstack(chunks).astype(np.float16, copy=False)


def load_model_for_worker(rank: int, cfg: dict[str, Any]) -> tuple[Any, str]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from sentence_transformers import SentenceTransformer

    if bool(cfg["cpu"]):
        device = "cpu"
    elif torch.cuda.is_available():
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    model = SentenceTransformer(
        cfg["model"],
        revision=cfg["revision"],
        trust_remote_code=True,
        device=device,
        cache_folder=cfg.get("cache_folder") or None,
    )
    model.max_seq_length = int(cfg["max_seq_length"])
    if bool(cfg["fp16"]) and device.startswith("cuda"):
        model = model.half()
    return model, device

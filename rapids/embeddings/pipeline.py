"""Shared embedding accumulation pipeline for all datasets.

Each dataset script provides dataset-specific record loading and field extraction;
this module handles everything else: the growing accumulator matrix, batched
encoding, mean-pooling, and L2-renormalisation.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def _ensure_capacity(
    cap: int,
    sum_mat: np.ndarray,
    cnt_arr: np.ndarray,
    n_current: int,
    n_new: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Double the accumulator arrays until they fit ``n_current + n_new`` rows."""
    need = n_current + n_new
    if need <= cap:
        return cap, sum_mat, cnt_arr
    emb_dim = sum_mat.shape[1]
    while cap < need:
        cap *= 2
    new_sum = np.zeros((cap, emb_dim), dtype=np.float32)
    new_sum[:n_current] = sum_mat[:n_current]
    new_cnt = np.zeros(cap, dtype=np.int32)
    new_cnt[:n_current] = cnt_arr[:n_current]
    return cap, new_sum, new_cnt


def run_embedding_pipeline(
    files: List[str],
    model,
    get_records: Callable[[str], Iterable[Any]],
    get_uid: Callable[[Any], Optional[Any]],
    get_text: Callable[[Any], str],
    batch_size: int,
    max_nodes: int = 0,
    stop_after_max_nodes: bool = True,
) -> Tuple[Dict[Any, int], np.ndarray, np.ndarray, Dict]:
    """Accumulate per-user mean-pooled embeddings across all files.

    Parameters
    ----------
    files:
        Ordered list of file paths to process.
    model:
        A loaded ``SentenceTransformer`` instance.
    get_records:
        Callable that takes a file path and returns an iterable of raw records.
    get_uid:
        Callable that extracts the user identity key (int or str) from a record.
        Return ``None`` to skip the record.
    get_text:
        Callable that extracts the text string to embed from a record.
        Return an empty string to skip the record.
    batch_size:
        Encoder batch size.
    max_nodes:
        Maximum number of unique users to admit (0 = no limit).
    stop_after_max_nodes:
        Stop reading further files once the cap is reached.

    Returns
    -------
    uid_to_row : dict
        Maps each admitted uid key to its row index in ``sum_mat`` / ``cnt_arr``.
    sum_mat : np.ndarray  [N, D]  float32
        Accumulated (unnormalised) embedding sums; valid rows are ``[:N]``.
    cnt_arr : np.ndarray  [N]  int32
        Number of posts accumulated per user.
    stats : dict
        Counters for logging / metadata.
    """
    t0 = time.time()
    emb_dim = model.get_sentence_embedding_dimension()

    uid_to_row: Dict[Any, int] = {}
    cap = 1 << 16
    sum_mat = np.zeros((cap, emb_dim), dtype=np.float32)
    cnt_arr = np.zeros(cap, dtype=np.int32)

    total_posts = 0
    total_items = 0
    total_missing_uid = 0
    total_empty_text = 0
    total_skip_new_uid = 0

    for file_idx, fpath in enumerate(files, start=1):
        ft = time.time()
        print(f"[{file_idx}/{len(files)}] loading {os.path.basename(fpath)}", flush=True)
        try:
            records = list(get_records(fpath))
        except Exception as exc:
            print(f"[{file_idx}/{len(files)}] ERROR {os.path.basename(fpath)}: {exc}", flush=True)
            continue
        if not records:
            print(f"[{file_idx}/{len(files)}] empty {os.path.basename(fpath)}", flush=True)
            continue

        texts: List[str] = []
        row_uids: List[Any] = []
        file_missing = 0
        file_empty = 0
        file_skip_new = 0

        for record in records:
            uid = get_uid(record)
            if uid is None:
                file_missing += 1
                continue
            text = get_text(record)
            if not text:
                file_empty += 1
                continue
            if uid not in uid_to_row:
                if max_nodes > 0 and len(uid_to_row) >= max_nodes:
                    file_skip_new += 1
                    continue
                cap, sum_mat, cnt_arr = _ensure_capacity(cap, sum_mat, cnt_arr, len(uid_to_row), 1)
                uid_to_row[uid] = len(uid_to_row)
            texts.append(text)
            row_uids.append(uid)

        total_items += len(records)
        total_missing_uid += file_missing
        total_empty_text += file_empty
        total_skip_new_uid += file_skip_new

        if not texts:
            continue

        unique_new = sum(1 for u in row_uids if cnt_arr[uid_to_row[u]] == 0)
        row_idx = np.fromiter((uid_to_row[u] for u in row_uids), dtype=np.int64, count=len(row_uids))

        embs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

        np.add.at(sum_mat, row_idx, embs)
        np.add.at(cnt_arr, row_idx, 1)

        total_posts += len(texts)
        dt = time.time() - ft
        print(
            f"[{file_idx}/{len(files)}] {os.path.basename(fpath)} "
            f"records={len(records):,} embedded={len(texts):,} "
            f"new_users={unique_new:,} users={len(uid_to_row):,} "
            f"skip_uid={file_missing:,} skip_empty={file_empty:,} skip_new={file_skip_new:,} "
            f"file={dt:.1f}s total={(time.time()-t0)/60:.1f}m",
            flush=True,
        )

        if stop_after_max_nodes and max_nodes > 0 and len(uid_to_row) >= max_nodes:
            print(
                f"Reached max_nodes={max_nodes:,} after file {file_idx}/{len(files)}; "
                "stopping (--stop_after_max_nodes is set).",
                flush=True,
            )
            break

    stats = {
        "total_items": total_items,
        "total_posts_embedded": total_posts,
        "total_missing_uid": total_missing_uid,
        "total_empty_text": total_empty_text,
        "total_skip_new_uid": total_skip_new_uid,
        "elapsed_min": (time.time() - t0) / 60,
    }
    return uid_to_row, sum_mat, cnt_arr, stats


def finalize_embeddings(
    uid_to_row: Dict[Any, int],
    sum_mat: np.ndarray,
    cnt_arr: np.ndarray,
    max_nodes: int = 0,
) -> Tuple[List[Any], torch.Tensor, np.ndarray]:
    """Convert raw accumulators to a mean-pooled, L2-normalised embedding tensor.

    Returns
    -------
    keys : list
        Uid keys in row order (same ordering as ``uid_to_row``).
    meanpool : torch.Tensor  [N, D]  float32
    counts : np.ndarray  [N]  int64
    """
    n = len(uid_to_row)
    if max_nodes > 0:
        if n > max_nodes:
            raise RuntimeError(f"Embedding cap failed: requested {max_nodes:,}, got {n:,}")
        if n < max_nodes:
            print(f"[WARN] requested max_nodes={max_nodes:,}, only {n:,} users admitted.", flush=True)

    keys: List[Any] = [None] * n
    for k, row in uid_to_row.items():
        keys[row] = k

    counts = cnt_arr[:n].astype(np.int64)
    denom = np.maximum(counts, 1).astype(np.float32)[:, None]
    meanpool = sum_mat[:n] / denom
    norms = np.linalg.norm(meanpool, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    meanpool = meanpool / norms

    return keys, torch.from_numpy(meanpool), counts

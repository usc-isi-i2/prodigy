import argparse
import glob
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/*/*.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = "data/data/covid19_twitter/embeddings/user_embeddings_minilm.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-user pooled text embeddings from raw covid19_twitter JSON files."
    )
    p.add_argument("--json_glob", default=DEFAULT_JSON_GLOB)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument(
        "--checkpoint_path",
        default="data/data/covid19_twitter/embeddings/user_embeddings_minilm.checkpoint.pkl",
    )
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--resume", action="store_true", default=False)
    return p.parse_args()


def normalize_handle(handle):
    if handle is None:
        return None
    s = str(handle).strip().lower()
    return s if s and s not in {"nan", "none", "<na>"} else None


def load_json_items(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("statuses"), list):
                return obj["statuses"]
            if isinstance(obj.get("data"), list):
                return obj["data"]
            return [obj]
        return []
    except json.JSONDecodeError:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items


def get_tweet_text(tweet: dict) -> str:
    extended = tweet.get("extended_tweet") or {}
    retweeted = tweet.get("retweeted_status") or {}
    retweeted_ext = retweeted.get("extended_tweet") or {}
    for candidate in (
        extended.get("full_text"),
        tweet.get("full_text"),
        tweet.get("text"),
        retweeted_ext.get("full_text"),
        retweeted.get("full_text"),
        retweeted.get("text"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return ""


def build_text(tweet: dict) -> str:
    parts = []
    text = get_tweet_text(tweet)
    if text:
        parts.append(text)
    user = tweet.get("user") or {}
    desc = user.get("description")
    if desc and str(desc).strip():
        desc = str(desc).strip()
        if not parts or parts[0] != desc:
            parts.append(desc)
    return " ".join(parts)[:512] if parts else ""


def save_checkpoint(path, user_sum, user_max, user_count, files_done):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "user_sum": dict(user_sum),
                "user_max": dict(user_max),
                "user_count": dict(user_count),
                "files_done": list(files_done),
            },
            f,
        )
    os.replace(tmp, path)
    print(f"  [CHECKPOINT] saved {len(files_done):,} files -> {path}", flush=True)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()
    start_time = time.time()
    files = sorted(glob.glob(args.json_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.json_glob}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_len
    emb_dim = model.get_sentence_embedding_dimension()

    print(f"Embedding model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Files: {len(files)}")
    print(f"Output: {args.out}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Resume: {args.resume}")

    checkpoint = load_checkpoint(args.checkpoint_path) if args.resume else None
    if checkpoint:
        user_sum = defaultdict(lambda: np.zeros(emb_dim, dtype=np.float64), checkpoint.get("user_sum", {}))
        user_max = defaultdict(lambda: np.full(emb_dim, -np.inf, dtype=np.float32), checkpoint.get("user_max", {}))
        user_count = defaultdict(int, checkpoint.get("user_count", {}))
        files_done = set(checkpoint.get("files_done", []))
        print(
            f"Resumed from checkpoint: files_done={len(files_done):,} users={len(user_sum):,} "
            f"posts={sum(user_count.values()):,}",
            flush=True,
        )
    else:
        user_sum = defaultdict(lambda: np.zeros(emb_dim, dtype=np.float64))
        user_max = defaultdict(lambda: np.full(emb_dim, -np.inf, dtype=np.float32))
        user_count = defaultdict(int)
        files_done = set()

    processed_in_run = 0
    for i, fpath in enumerate(files, start=1):
        if fpath in files_done:
            print(f"[{i}/{len(files)}] Skipping {os.path.basename(fpath)} (already checkpointed)", flush=True)
            continue
        if args.max_files > 0 and processed_in_run >= args.max_files:
            print(f"Reached max_files={args.max_files}; stopping this run.", flush=True)
            break

        file_start = time.time()
        print(f"[{i}/{len(files)}] Loading {os.path.basename(fpath)}", flush=True)
        try:
            items = load_json_items(fpath)
        except Exception as exc:
            print(f"  [ERROR] Skipping {os.path.basename(fpath)}: {exc}", flush=True)
            continue

        if not items:
            print("  [SKIP] empty file", flush=True)
            continue

        texts = []
        handles = []
        for tweet in items:
            user = tweet.get("user") or {}
            handle = normalize_handle(user.get("screen_name"))
            if not handle:
                continue
            text = build_text(tweet)
            if not text:
                continue
            handles.append(handle)
            texts.append(text)

        if not texts:
            print(f"  [SKIP] no non-empty texts out of {len(items):,} tweets", flush=True)
            continue

        print(
            f"  tweets={len(items):,} texts_to_embed={len(texts):,}",
            flush=True,
        )

        embs = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for handle, emb in zip(handles, embs):
            user_sum[handle] += emb.astype(np.float64)
            user_max[handle] = np.maximum(user_max[handle], emb.astype(np.float32))
            user_count[handle] += 1

        files_done.add(fpath)
        processed_in_run += 1
        elapsed = time.time() - file_start
        total_elapsed = time.time() - start_time
        print(
            "  [DONE] "
            f"file_time={elapsed:.1f}s total_time={total_elapsed/60:.1f}m "
            f"cumulative_users={len(user_sum):,} cumulative_posts={sum(user_count.values()):,}",
            flush=True,
        )
        if args.checkpoint_every > 0 and processed_in_run % args.checkpoint_every == 0:
            save_checkpoint(args.checkpoint_path, user_sum, user_max, user_count, files_done)

    save_checkpoint(args.checkpoint_path, user_sum, user_max, user_count, files_done)

    handles = sorted(user_sum.keys())
    n = len(handles)
    meanpool = torch.zeros(n, emb_dim, dtype=torch.float32)
    maxpool = torch.zeros(n, emb_dim, dtype=torch.float32)

    for idx, handle in enumerate(handles):
        meanpool[idx] = torch.from_numpy((user_sum[handle] / user_count[handle]).astype(np.float32))
        maxpool[idx] = torch.from_numpy(user_max[handle])

    out_obj = {
        "handles": handles,
        "meanpool": meanpool,
        "maxpool": maxpool,
        "counts": {handle: int(user_count[handle]) for handle in handles},
        "model": args.model,
    }
    torch.save(out_obj, args.out)

    meta = {
        "json_glob": args.json_glob,
        "files_count": len(files),
        "model": args.model,
        "embedding_dim": int(emb_dim),
        "users": int(n),
        "total_posts_embedded": int(sum(user_count.values())),
    }
    with open(args.out.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings: {args.out}")
    print(f"users={n:,}, dim={emb_dim}")


if __name__ == "__main__":
    main()

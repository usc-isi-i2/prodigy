"""Build per-user mean-pooled text embeddings from covid19_twitter JSON files."""
import argparse
import glob
import json
import os
import shlex
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rapids.embeddings.pipeline import finalize_embeddings, run_embedding_pipeline
from rapids.loaders.json_loader import load_json_items
from rapids.utils import normalize_handle, normalize_user_id

DEFAULT_JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = "/scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm.pt"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-user (userid) pooled text embeddings from covid19_twitter JSON files."
    )
    p.add_argument("--json_glob", default=DEFAULT_JSON_GLOB)
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


def _get_uid(tweet: dict):
    user = tweet.get("user") or {}
    return normalize_user_id(user.get("id"))


def _get_text(tweet: dict) -> str:
    """Use retweeted-status text for pure RTs; own text otherwise."""
    def _status_text(status):
        if not status:
            return ""
        ext = status.get("extended_tweet") or {}
        for candidate in (ext.get("full_text"), status.get("full_text"), status.get("text")):
            if candidate:
                s = str(candidate).strip()
                if s:
                    return s
        return ""

    retweeted = tweet.get("retweeted_status")
    return _status_text(retweeted) if retweeted else _status_text(tweet)


def main():
    args = parse_args()
    command = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])

    files = sorted(glob.glob(args.json_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.json_glob}")

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

    # Track handles in insertion order alongside uid_to_row
    _handles: list = []

    def _get_records_with_handle_tracking(fpath: str):
        items = load_json_items(fpath)
        # Attach screen_name so we can capture it during get_uid calls
        for item in items:
            item["_fpath"] = fpath  # not needed but kept for symmetry
        return items

    # We need handles aligned with uid_to_row — intercept uid assignment
    _uid_seen: set = set()

    def _get_uid_tracking(tweet: dict):
        uid = _get_uid(tweet)
        if uid is not None and uid not in _uid_seen:
            _uid_seen.add(uid)
            user = tweet.get("user") or {}
            _handles.append(normalize_handle(user.get("screen_name")))
        return uid

    uid_to_row, sum_mat, cnt_arr, stats = run_embedding_pipeline(
        files=files,
        model=model,
        get_records=load_json_items,
        get_uid=_get_uid_tracking,
        get_text=_get_text,
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        stop_after_max_nodes=args.stop_after_max_nodes,
    )

    keys, meanpool, counts = finalize_embeddings(uid_to_row, sum_mat, cnt_arr, args.max_nodes)
    user_ids = np.array(keys, dtype=np.int64)

    torch.save({
        "user_ids": user_ids,
        "handles": _handles,
        "meanpool": meanpool,
        "counts": counts,
        "model": args.model,
    }, args.out)

    meta = {
        "json_glob": args.json_glob,
        "files_count": len(files),
        "model": args.model,
        "embedding_dim": int(emb_dim),
        "users": int(len(user_ids)),
        "max_nodes": int(args.max_nodes),
        "stop_after_max_nodes": bool(args.stop_after_max_nodes),
        "command": command,
        "max_seq_len": args.max_seq_len,
        "fp16": bool(args.fp16),
        **stats,
    }
    with open(args.out.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {args.out} users={len(user_ids):,} dim={emb_dim} "
          f"posts={stats['total_posts_embedded']:,} wall={stats['elapsed_min']:.1f}m")


if __name__ == "__main__":
    main()

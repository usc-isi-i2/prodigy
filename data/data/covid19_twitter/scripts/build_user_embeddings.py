import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import orjson
def _loads(s): return orjson.loads(s)


DEFAULT_JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = "data/data/covid19_twitter/embeddings/user_embeddings_minilm.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_glob", default=DEFAULT_JSON_GLOB)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max_seq_len", type=int, default=64)
    p.add_argument("--fp16", action="store_true", default=True)
    return p.parse_args()


def normalize_user_id(user_id):
    if user_id is None:
        return None
    try:
        return int(user_id)
    except Exception:
        return None


def normalize_handle(h):
    if h is None:
        return None
    s = str(h).strip().lower()
    return s if s and s not in {"nan", "none", "<na>"} else None


def load_json_items(path):
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.strip():
        return []
    try:
        obj = _loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("statuses", "data"):
                if isinstance(obj.get(k), list):
                    return obj[k]
            return [obj]
        return []
    except Exception:
        # JSONL fallback
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(_loads(line))
            except Exception:
                pass
        return items


def _status_text(status):
    """Extract the text body from a status dict, preferring extended/full_text."""
    if not status:
        return ""
    ext = status.get("extended_tweet") or {}
    for c in (ext.get("full_text"), status.get("full_text"), status.get("text")):
        if c:
            s = str(c).strip()
            if s:
                return s
    return ""


def build_text(tweet):
    """
    Original tweet  -> user's own text
    Pure retweet    -> original (retweeted_status) text   [endorsement signal]
    Quote tweet     -> user's added comment (= the tweet's own text)
    """
    retweeted = tweet.get("retweeted_status")
    if retweeted:
        # Pure RTs: tweet text is just "RT @user: ..."; use the source text instead.
        return _status_text(retweeted)

    # Quote tweet OR original tweet -> the user's own composed text.
    return _status_text(tweet)


def main():
    args = parse_args()
    t0 = time.time()

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
    print(f"Input glob={args.json_glob}")
    print(f"Output={args.out}")

    # Dense accumulators, grown as new users appear.
    uid_to_row = {}
    handles = []           # parallel to uid_to_row rows
    sum_rows = []          # list of np.ndarray(emb_dim, float32) blocks
    counts = []            # list of int

    # Use a single growing float32 matrix doubled on demand.
    cap = 1 << 16
    sum_mat = np.zeros((cap, emb_dim), dtype=np.float32)
    cnt_arr = np.zeros(cap, dtype=np.int32)

    def ensure_capacity(n_new):
        nonlocal cap, sum_mat, cnt_arr
        need = len(uid_to_row) + n_new
        if need <= cap:
            return
        while cap < need:
            cap *= 2
        new_sum = np.zeros((cap, emb_dim), dtype=np.float32)
        new_sum[: len(uid_to_row)] = sum_mat[: len(uid_to_row)]
        sum_mat = new_sum
        new_cnt = np.zeros(cap, dtype=np.int32)
        new_cnt[: len(uid_to_row)] = cnt_arr[: len(uid_to_row)]
        cnt_arr = new_cnt

    total_posts = 0
    total_items = 0
    total_missing_uid = 0
    total_empty_text = 0
    for i, fpath in enumerate(files, start=1):
        ft = time.time()
        print(f"[{i}/{len(files)}] loading {os.path.basename(fpath)}", flush=True)
        try:
            items = load_json_items(fpath)
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {os.path.basename(fpath)}: {e}", flush=True)
            continue
        if not items:
            print(f"[{i}/{len(files)}] empty {os.path.basename(fpath)}", flush=True)
            continue

        texts = []
        rows = []  # row index in sum_mat for each text
        file_missing_uid = 0
        file_empty_text = 0

        # First pass: collect texts and assign row ids (cheap, no GPU).
        new_uids = []
        for tw in items:
            user = tw.get("user") or {}
            uid = normalize_user_id(user.get("id"))
            if uid is None:
                file_missing_uid += 1
                continue
            text = build_text(tw)
            if not text:
                file_empty_text += 1
                continue
            row = uid_to_row.get(uid)
            if row is None:
                new_uids.append((uid, normalize_handle(user.get("screen_name"))))
            texts.append(text)
            rows.append(uid)  # temporarily store uid; resolve after capacity grow

        if not texts:
            print(f"[{i}/{len(files)}] no texts in {len(items):,} tweets", flush=True)
            total_items += len(items)
            total_missing_uid += file_missing_uid
            total_empty_text += file_empty_text
            continue

        unique_new_user_count = len({uid for uid, _ in new_uids})
        if new_uids:
            ensure_capacity(len(new_uids))
            for uid, handle in new_uids:
                if uid not in uid_to_row:
                    uid_to_row[uid] = len(uid_to_row)
                    handles.append(handle)

        row_idx = np.fromiter((uid_to_row[u] for u in rows), dtype=np.int64, count=len(rows))

        # GPU encode (already L2-normalized).
        embs = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

        # Vectorized scatter-add.
        np.add.at(sum_mat, row_idx, embs)
        np.add.at(cnt_arr, row_idx, 1)

        total_posts += len(texts)
        total_items += len(items)
        total_missing_uid += file_missing_uid
        total_empty_text += file_empty_text
        dt = time.time() - ft
        print(
            f"[{i}/{len(files)}] {os.path.basename(fpath)} "
            f"tweets={len(items):,} embedded={len(texts):,} "
            f"new_users={unique_new_user_count:,} users={len(uid_to_row):,} "
            f"skip_uid={file_missing_uid:,} skip_empty={file_empty_text:,} "
            f"file={dt:.1f}s "
            f"total={ (time.time()-t0)/60:.1f}m",
            flush=True,
        )

    n = len(uid_to_row)
    user_ids = np.empty(n, dtype=np.int64)
    for uid, row in uid_to_row.items():
        user_ids[row] = uid

    counts_final = cnt_arr[:n].astype(np.int64)
    # Mean pool, guarding against zero (shouldn't happen).
    denom = np.maximum(counts_final, 1).astype(np.float32)[:, None]
    meanpool = sum_mat[:n] / denom
    # Re-normalize the mean to unit length (common for averaged sentence embeddings).
    norms = np.linalg.norm(meanpool, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    meanpool = meanpool / norms

    out_obj = {
        "user_ids": user_ids,
        "handles": handles,
        "meanpool": torch.from_numpy(meanpool),
        "counts": counts_final,
        "model": args.model,
    }
    torch.save(out_obj, args.out)

    meta = {
        "json_glob": args.json_glob,
        "files_count": len(files),
        "model": args.model,
        "embedding_dim": int(emb_dim),
        "users": int(n),
        "total_posts_embedded": int(total_posts),
        "max_seq_len": args.max_seq_len,
        "fp16": bool(args.fp16),
    }
    with open(args.out.replace(".pt", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {args.out} users={n:,} dim={emb_dim} posts={total_posts:,} "
          f"wall={(time.time()-t0)/60:.1f}m")
    print(
        "Summary: "
        f"tweets_seen={total_items:,} embedded={total_posts:,} "
        f"skip_uid={total_missing_uid:,} skip_empty={total_empty_text:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

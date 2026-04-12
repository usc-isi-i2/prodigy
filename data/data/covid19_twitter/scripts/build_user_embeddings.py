import argparse
import glob
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import orjson as _json
def json_loads(s): return _json.loads(s)


DEFAULT_JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = "data/data/covid19_twitter/embeddings/user_embeddings_minilm.pt"
TEXT_FIELDS_SUPPORTED = ("tweet_text", "retweet_text", "quote_text", "bio")
DEFAULT_TEXT_FIELDS = "tweet_text,retweet_text,quote_text"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_glob", default=DEFAULT_JSON_GLOB)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max_seq_len", type=int, default=48)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--text_fields", default=DEFAULT_TEXT_FIELDS)
    return p.parse_args()


def parse_text_fields(spec):
    fields = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [f for f in fields if f not in TEXT_FIELDS_SUPPORTED]
    if bad:
        raise ValueError(f"Unsupported text fields: {bad}")
    return fields


def load_json_items(path):
    with open(path, "rb") as f:
        data = f.read()
    if not data.strip():
        return []
    try:
        obj = json_loads(data)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("statuses", "data"):
                if isinstance(obj.get(k), list):
                    return obj[k]
            return [obj]
        return []
    except Exception:
        items = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json_loads(line))
            except Exception:
                pass
        return items


def get_status_text(status):
    ext = status.get("extended_tweet") or {}
    for c in (ext.get("full_text"), status.get("full_text"), status.get("text")):
        if c:
            s = str(c).strip()
            if s:
                return s
    return ""


def build_text(tweet, text_fields):
    parts = []
    seen = set()
    rt = tweet.get("retweeted_status") or {}
    qt = tweet.get("quoted_status") or {}
    user = tweet.get("user") or {}
    vals = {
        "tweet_text": get_status_text(tweet),
        "retweet_text": get_status_text(rt) if rt else "",
        "quote_text": get_status_text(qt) if qt else "",
        "bio": str(user.get("description") or "").strip(),
    }
    for f in text_fields:
        v = vals.get(f, "")
        if v and v not in seen:
            parts.append(v)
            seen.add(v)
    return " ".join(parts)


def main():
    args = parse_args()
    t0 = time.time()
    text_fields = parse_text_fields(args.text_fields)

    files = sorted(glob.glob(args.json_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(args.json_glob)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_len
    if args.fp16 and args.device.startswith("cuda"):
        model.half()
    emb_dim = model.get_sentence_embedding_dimension()

    print(f"model={args.model} device={args.device} dim={emb_dim}")
    print(f"files={len(files)} batch={args.batch_size} seq={args.max_seq_len} fp16={args.fp16}")
    print(f"text_fields={text_fields}")

    # Two-pass-free streaming reduction:
    # Map user_id -> contiguous row index on the fly. Accumulate into dense
    # float32 matrices that we grow in chunks.
    uid_to_row = {}
    handles = {}
    CHUNK = 200_000
    sum_mat = np.zeros((CHUNK, emb_dim), dtype=np.float32)
    cnt_vec = np.zeros(CHUNK, dtype=np.int32)
    n_users = 0

    total_tweets = 0
    total_embedded = 0

    for i, fpath in enumerate(files, 1):
        fstart = time.time()
        try:
            items = load_json_items(fpath)
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {os.path.basename(fpath)}: {e}", flush=True)
            continue
        if not items:
            continue
        total_tweets += len(items)

        texts = []
        uids = []
        for tw in items:
            user = tw.get("user") or {}
            uid = user.get("id")
            if uid is None:
                continue
            try:
                uid = int(uid)
            except Exception:
                continue
            text = build_text(tw, text_fields)
            if not text:
                continue
            uids.append(uid)
            texts.append(text)
            sn = user.get("screen_name")
            if sn and uid not in handles:
                handles[uid] = str(sn).strip().lower()

        if not texts:
            continue

        embs = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

        # Map uids to row indices, growing matrices as needed.
        rows = np.empty(len(uids), dtype=np.int64)
        for j, uid in enumerate(uids):
            r = uid_to_row.get(uid)
            if r is None:
                r = n_users
                uid_to_row[uid] = r
                n_users += 1
                if n_users > sum_mat.shape[0]:
                    new_size = max(sum_mat.shape[0] * 2, n_users + CHUNK)
                    new_sum = np.zeros((new_size, emb_dim), dtype=np.float32)
                    new_cnt = np.zeros(new_size, dtype=np.int32)
                    new_sum[: sum_mat.shape[0]] = sum_mat
                    new_cnt[: cnt_vec.shape[0]] = cnt_vec
                    sum_mat = new_sum
                    cnt_vec = new_cnt
            rows[j] = r

        # Vectorized scatter-add (handles duplicate rows correctly).
        np.add.at(sum_mat, rows, embs)
        np.add.at(cnt_vec, rows, 1)
        total_embedded += len(texts)

        ftime = time.time() - fstart
        print(
            f"[{i}/{len(files)}] {os.path.basename(fpath)} "
            f"tweets={len(items):,} embedded={len(texts):,} "
            f"users={n_users:,} ftime={ftime:.1f}s "
            f"total={ (time.time()-t0)/60:.1f}m",
            flush=True,
        )

    # Finalize
    sum_mat = sum_mat[:n_users]
    cnt_vec = cnt_vec[:n_users]
    meanpool = sum_mat / np.maximum(cnt_vec, 1)[:, None]

    # Sort by user_id for deterministic output
    user_ids_arr = np.empty(n_users, dtype=np.int64)
    for uid, r in uid_to_row.items():
        user_ids_arr[r] = uid
    order = np.argsort(user_ids_arr)
    user_ids_sorted = user_ids_arr[order]
    meanpool = meanpool[order]
    cnt_sorted = cnt_vec[order]
    handle_list = [handles.get(int(u)) for u in user_ids_sorted]

    out_obj = {
        "user_ids": user_ids_sorted,
        "handles": handle_list,
        "meanpool": torch.from_numpy(meanpool),
        "counts": torch.from_numpy(cnt_sorted.astype(np.int64)),
        "model": args.model,
    }
    torch.save(out_obj, args.out)

    print(
        f"DONE users={n_users:,} tweets_seen={total_tweets:,} "
        f"embedded={total_embedded:,} time={(time.time()-t0)/60:.1f}m"
    )


if __name__ == "__main__":
    main()
from collections import defaultdict
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer("intfloat/e5-base-v2")
embed_dim = 768
batch_size = 512
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

PROCESSED_FILE = checkpoint_dir / "processed_files.json"
SAVE_EVERY = 1          # save checkpoint every N files
SHARD_MAX_ROWS = 500_000  # max users per shard file


def prefix_texts(texts, prefix="passage: "):
    return [prefix + t for t in texts]


def _shard_path(shard_idx):
    return checkpoint_dir / f"state_{shard_idx:04d}.npz"


def save_checkpoint(tweet_sum, tweet_max, tweet_count,
                    bio_sum, bio_max, bio_count, user_bios,
                    processed_files):
    """Save all accumulators across sharded .npz files + metadata JSON."""
    all_uids = sorted(set(tweet_sum.keys()) | set(bio_sum.keys()))
    n_shards = (len(all_uids) + SHARD_MAX_ROWS - 1) // SHARD_MAX_ROWS

    # Remove any old shards beyond what we'll write now
    for old in checkpoint_dir.glob("state_*.npz"):
        old.unlink()

    for shard_idx in range(n_shards):
        start = shard_idx * SHARD_MAX_ROWS
        end = min(start + SHARD_MAX_ROWS, len(all_uids))
        shard_uids = all_uids[start:end]
        uid_array = np.array(shard_uids)

        t_sum = np.array([tweet_sum.get(u, np.zeros(embed_dim)) for u in shard_uids])
        t_max = np.array([tweet_max.get(u, np.full(embed_dim, -np.inf)) for u in shard_uids])
        t_cnt = np.array([tweet_count.get(u, 0) for u in shard_uids])

        b_sum = np.array([bio_sum.get(u, np.zeros(embed_dim)) for u in shard_uids])
        b_max = np.array([bio_max.get(u, np.full(embed_dim, -np.inf)) for u in shard_uids])
        b_cnt = np.array([bio_count.get(u, 0) for u in shard_uids])

        np.savez_compressed(
            _shard_path(shard_idx),
            uids=uid_array,
            tweet_sum=t_sum, tweet_max=t_max, tweet_count=t_cnt,
            bio_sum=b_sum, bio_max=b_max, bio_count=b_cnt,
        )

    # Save processed files + user_bios as JSON
    meta = {
        "processed_files": processed_files,
        "n_shards": n_shards,
        "shard_max_rows": SHARD_MAX_ROWS,
        "total_users": len(all_uids),
        "user_bios": {str(k): list(v) for k, v in user_bios.items()},
    }
    with open(PROCESSED_FILE, "w") as f:
        json.dump(meta, f)

    print(f"  [checkpoint] Saved {len(all_uids)} users across {n_shards} shard(s), "
          f"{len(processed_files)} files processed")


def load_checkpoint():
    """Restore accumulators and processed file list from sharded files."""
    tweet_sum = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float64))
    tweet_max = defaultdict(lambda: np.full(embed_dim, -np.inf, dtype=np.float64))
    tweet_count = defaultdict(int)
    bio_sum = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float64))
    bio_max = defaultdict(lambda: np.full(embed_dim, -np.inf, dtype=np.float64))
    bio_count = defaultdict(int)
    user_bios = defaultdict(set)
    processed_files = []

    if not PROCESSED_FILE.exists():
        print("No checkpoint found, starting fresh.")
        return (tweet_sum, tweet_max, tweet_count,
                bio_sum, bio_max, bio_count,
                user_bios, processed_files)

    with open(PROCESSED_FILE) as f:
        meta = json.load(f)

    processed_files = meta["processed_files"]
    n_shards = meta["n_shards"]

    for k, v in meta["user_bios"].items():
        try:
            key = int(k)
        except ValueError:
            key = k
        user_bios[key] = set(v)

    total_loaded = 0
    for shard_idx in range(n_shards):
        path = _shard_path(shard_idx)
        if not path.exists():
            print(f"  WARNING: missing shard {path}, skipping")
            continue
        data = np.load(path)
        uids = data["uids"]
        for i, uid in enumerate(uids):
            uid = int(uid) if np.issubdtype(type(uid), np.integer) else uid
            if data["tweet_count"][i] > 0:
                tweet_sum[uid] = data["tweet_sum"][i].astype(np.float64)
                tweet_max[uid] = data["tweet_max"][i].astype(np.float64)
                tweet_count[uid] = int(data["tweet_count"][i])
            if data["bio_count"][i] > 0:
                bio_sum[uid] = data["bio_sum"][i].astype(np.float64)
                bio_max[uid] = data["bio_max"][i].astype(np.float64)
                bio_count[uid] = int(data["bio_count"][i])
        total_loaded += len(uids)

    print(f"  Resumed: {len(processed_files)} files done, "
          f"{total_loaded} users from {n_shards} shard(s), "
          f"{sum(tweet_count.values()):,} tweets, "
          f"{sum(bio_count.values()):,} bios")

    return (tweet_sum, tweet_max, tweet_count,
            bio_sum, bio_max, bio_count,
            user_bios, processed_files)


# ── Load or initialize state ──
(tweet_sum, tweet_max, tweet_count,
 bio_sum, bio_max, bio_count,
 user_bios, processed_files) = load_checkpoint()

processed_set = set(processed_files)

base_dir = "/project2/ll_774_951/TwitterInfoOpsIU/YYYY_MM/"
files = list(Path(base_dir).rglob("*/*/*.csv.gz"))
import random
n = len(files)
files = random.sample(files, n)
print(f"Found {len(files)} files total, {len(processed_set)} already done")

total_rows = sum(tweet_count.values())
total_users = set(tweet_sum.keys())
t0 = time.time()
files_since_save = 0

for i, f in enumerate(files):
    if str(f) in processed_set:
        continue

    file_start = time.time()
    print(f"\n[{i+1}/{len(files)}] Reading ... {f.name}")
    df = pd.read_csv(f, compression="gzip", low_memory=False,
                     usecols=["userid", "tweet_text", "user_profile_description"])
    print(f"  Loaded {len(df):,} rows in {time.time() - file_start:.1f}s")

    # collect unique bios
    bio_before = sum(len(v) for v in user_bios.values())
    for uid, bio in zip(df["userid"], df["user_profile_description"]):
        if pd.notna(bio):
            user_bios[uid].add(bio.strip())
    bio_after = sum(len(v) for v in user_bios.values())
    print(f"  Collected {bio_after - bio_before} new unique bios (total: {bio_after})")

    # embed tweets
    texts = df["tweet_text"].fillna("").tolist()
    uids = df["userid"].values
    file_users = set(uids)
    total_users.update(file_users)

    n_batches = (len(df) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(df), batch_size),
                      total=n_batches,
                      desc=f"  Embedding tweets",
                      unit="batch"):
        end = min(start + batch_size, len(df))
        batch_texts = prefix_texts(texts[start:end])
        batch_uids = uids[start:end]
        embeddings = model.encode(batch_texts, normalize_embeddings=True,
                                  show_progress_bar=False)
        for uid, emb in zip(batch_uids, embeddings):
            tweet_sum[uid] += emb
            tweet_max[uid] = np.maximum(tweet_max[uid], emb)
            tweet_count[uid] += 1

    # embed bios for users in this file (incremental)
    file_bio_items = [(uid, bio) for uid in file_users
                      if uid in user_bios
                      for bio in user_bios[uid]]
    users_with_new_bios = {uid for uid in file_users if uid in user_bios}
    for uid in users_with_new_bios:
        bio_sum[uid] = np.zeros(embed_dim, dtype=np.float64)
        bio_max[uid] = np.full(embed_dim, -np.inf, dtype=np.float64)
        bio_count[uid] = 0
    if file_bio_items:
        for start in range(0, len(file_bio_items), batch_size):
            end = min(start + batch_size, len(file_bio_items))
            batch = file_bio_items[start:end]
            embeddings = model.encode(prefix_texts([b for _, b in batch]),
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
            for (uid, _), emb in zip(batch, embeddings):
                bio_sum[uid] += emb
                bio_max[uid] = np.maximum(bio_max[uid], emb)
                bio_count[uid] += 1

    total_rows += len(df)
    elapsed = time.time() - t0

    processed_files.append(str(f))
    processed_set.add(str(f))
    files_since_save += 1

    print(f"  {len(df):,} rows, {len(file_users):,} users | "
          f"file took {time.time() - file_start:.1f}s | "
          f"cumulative: {total_rows:,} rows, {len(total_users):,} users, {elapsed:.1f}s")

    if files_since_save >= SAVE_EVERY:
        save_checkpoint(tweet_sum, tweet_max, tweet_count,
                        bio_sum, bio_max, bio_count, user_bios,
                        processed_files)
        files_since_save = 0

# Final save
save_checkpoint(tweet_sum, tweet_max, tweet_count,
                bio_sum, bio_max, bio_count, user_bios,
                processed_files)

# Finalize
tweet_mean = {uid: tweet_sum[uid] / tweet_count[uid] for uid in tweet_sum}
bio_mean = {uid: bio_sum[uid] / bio_count[uid] for uid in bio_sum}

print(f"\nTweet embeddings: {len(tweet_mean)} users, {sum(tweet_count.values()):,} tweets")
print(f"Bio embeddings: {len(bio_mean)} users, {sum(bio_count.values()):,} bios")
print(f"All done in {time.time() - t0:.1f}s")
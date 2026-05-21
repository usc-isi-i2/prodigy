import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"
DEFAULT_WORKERS = min(10, os.cpu_count() or 1)
DEFAULT_BATCH_SIZE = 25


def normalize_user_id(user_id):
    if user_id is None:
        return None
    try:
        return int(user_id)
    except Exception:
        return None


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
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items


def batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def iter_user_ids(tweet):
    user = tweet.get("user") or {}
    rt = tweet.get("retweeted_status") or {}
    rt_user = rt.get("user") or {}
    quoted = tweet.get("quoted_status") or {}
    quoted_user = quoted.get("user") or {}

    candidates = [
        user.get("id"),
        rt_user.get("id") if rt else None,
        tweet.get("in_reply_to_user_id"),
        quoted_user.get("id") if quoted else None,
    ]
    for raw_user_id in candidates:
        user_id = normalize_user_id(raw_user_id)
        if user_id is not None:
            yield user_id

    entities = tweet.get("entities", {}) or {}
    mentions = entities.get("user_mentions", []) or []
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        user_id = normalize_user_id(mention.get("id"))
        if user_id is not None:
            yield user_id


def process_file_batch(paths):
    total_tweets = 0
    unique_ids = set()
    skipped = []

    for path in paths:
        try:
            items = load_json_items(path)
            total_tweets += len(items)
            for tweet in items:
                for user_id in iter_user_ids(tweet):
                    unique_ids.add(user_id)
        except Exception as exc:
            skipped.append((path, str(exc)))

    return total_tweets, unique_ids, skipped


def main():
    files = sorted(glob.glob(JSON_GLOB))
    print(f"Found {len(files)} files", flush=True)

    workers = int(os.environ.get("COUNT_USERS_WORKERS", DEFAULT_WORKERS))
    batch_size = int(os.environ.get("COUNT_USERS_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    total_tweets = 0
    unique_ids = set()
    start = time.time()
    done_files = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_file_batch, batch): batch
            for batch in batched(files, batch_size)
        }

        for future in as_completed(futures):
            batch = futures[future]
            batch_tweets, batch_user_ids, skipped = future.result()
            total_tweets += batch_tweets
            unique_ids.update(batch_user_ids)
            done_files += len(batch)

            for path, exc in skipped:
                print(f"  skipped {path}: {exc}", flush=True)

            elapsed = time.time() - start
            rate = done_files / elapsed if elapsed > 0 else 0
            eta = (len(files) - done_files) / rate if rate > 0 else 0
            print(
                f"[{done_files:>4}/{len(files)}] "
                f"tweets={total_tweets:>12,}  "
                f"unique_users={len(unique_ids):>10,}  "
                f"elapsed={elapsed/60:5.1f}m  "
                f"eta={eta/60:5.1f}m",
                flush=True,
            )

    print(f"\nFinal: {total_tweets:,} tweets, {len(unique_ids):,} unique user ids")
    print(f"Took {(time.time() - start)/60:.1f} minutes")


if __name__ == "__main__":
    main()

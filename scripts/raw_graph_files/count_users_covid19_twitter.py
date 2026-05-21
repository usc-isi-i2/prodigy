import glob
import json
import os
import time


JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"


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


files = sorted(glob.glob(JSON_GLOB))
print(f"Found {len(files)} files", flush=True)

total_tweets = 0
unique_ids = set()
start = time.time()

for i, path in enumerate(files, 1):
    try:
        items = load_json_items(path)
        total_tweets += len(items)
        for tweet in items:
            for user_id in iter_user_ids(tweet):
                unique_ids.add(user_id)
    except Exception as exc:
        print(f"  skipped {path}: {exc}", flush=True)

    if i % 10 == 0 or i == len(files):
        elapsed = time.time() - start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(files) - i) / rate if rate > 0 else 0
        print(
            f"[{i:>4}/{len(files)}] "
            f"tweets={total_tweets:>12,}  "
            f"unique_users={len(unique_ids):>10,}  "
            f"elapsed={elapsed/60:5.1f}m  "
            f"eta={eta/60:5.1f}m",
            flush=True,
        )

print(f"\nFinal: {total_tweets:,} tweets, {len(unique_ids):,} unique user ids")
print(f"Took {(time.time() - start)/60:.1f} minutes")

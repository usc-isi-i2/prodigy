import argparse
import glob
import json
import sys
import time


JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"


def normalize_user_id(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def load_json_items(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
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

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get("statuses"), list):
            return obj["statuses"]
        if isinstance(obj.get("data"), list):
            return obj["data"]
        return [obj]
    return []


def emit_header():
    print(
        "\t".join(
            [
                "file_index",
                "file_path",
                "rows_seen",
                "retweet_events",
                "unique_users",
                "unique_directed_edges",
                "new_retweet_events_in_file",
                "new_users_in_file",
                "new_unique_directed_edges_in_file",
                "elapsed_seconds",
            ]
        ),
        flush=True,
    )


def emit_row(
    file_index,
    path,
    rows_seen,
    retweet_events,
    unique_users,
    unique_edges,
    new_events,
    new_users,
    new_edges,
    elapsed_seconds,
):
    print(
        "\t".join(
            [
                str(file_index),
                path,
                str(rows_seen),
                str(retweet_events),
                str(unique_users),
                str(unique_edges),
                str(new_events),
                str(new_users),
                str(new_edges),
                f"{elapsed_seconds:.3f}",
            ]
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    files = sorted(glob.glob(JSON_GLOB))
    unique_users = set()
    unique_edges = set()
    rows_seen = 0
    retweet_events = 0
    start = time.time()

    emit_header()

    for file_index, path in enumerate(files, start=1):
        if args.max_files is not None and file_index > args.max_files:
            break

        file_new_users = 0
        file_new_edges = 0
        file_new_events = 0

        try:
            items = load_json_items(path)
            rows_seen += len(items)

            for tweet in items:
                user = tweet.get("user") or {}
                rt = tweet.get("retweeted_status") or {}
                rt_user = rt.get("user") or {}

                src = normalize_user_id(user.get("id"))
                dst = normalize_user_id(rt_user.get("id")) if rt else None
                if src is None or dst is None:
                    continue

                file_new_events += 1
                retweet_events += 1

                if src not in unique_users:
                    unique_users.add(src)
                    file_new_users += 1
                if dst not in unique_users:
                    unique_users.add(dst)
                    file_new_users += 1

                edge = (src, dst)
                if edge not in unique_edges:
                    unique_edges.add(edge)
                    file_new_edges += 1
        except Exception as exc:
            print(f"skipped\t{path}\t{exc}", file=sys.stderr, flush=True)

        emit_row(
            file_index=file_index,
            path=path,
            rows_seen=rows_seen,
            retweet_events=retweet_events,
            unique_users=len(unique_users),
            unique_edges=len(unique_edges),
            new_events=file_new_events,
            new_users=file_new_users,
            new_edges=file_new_edges,
            elapsed_seconds=time.time() - start,
        )

        if args.max_rows is not None and rows_seen >= args.max_rows:
            break
        if args.max_users is not None and len(unique_users) >= args.max_users:
            break


if __name__ == "__main__":
    main()

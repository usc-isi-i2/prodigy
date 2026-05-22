import argparse
import csv
import io
import glob
import re
import sys
import time


CSV_GLOB = "/project2/ll_774_951/midterm/*/*.csv"
START_PAT = re.compile(r"^\d{15,},")
csv.field_size_limit(sys.maxsize)


def normalize_user_id(value):
    if value is None:
        return None
    try:
        text = str(value).strip()
        if text == "":
            return None
        return int(text)
    except Exception:
        return None


def parse_record(text):
    return next(csv.reader(io.StringIO(text)))


def get_midterm_indices(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header = parse_record(f.readline())
    return header.index("userid"), header.index("rt_userid")


def read_midterm_rows(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header = parse_record(f.readline())
        expected_cols = len(header)

        current = []

        def flush():
            if not current:
                return None

            raw = "".join(current)
            try:
                row = parse_record(raw)
                if len(row) == expected_cols:
                    return row
            except Exception:
                pass

            repaired = " ".join(line.rstrip("\r\n") for line in current)
            row = parse_record(repaired)
            if len(row) != expected_cols:
                raise ValueError(f"bad row: got {len(row)} cols, expected {expected_cols}")
            return row

        for line in f:
            if START_PAT.match(line):
                row = flush()
                if row is not None:
                    yield row
                current = [line]
            else:
                current.append(line)

        row = flush()
        if row is not None:
            yield row


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

    files = sorted(glob.glob(CSV_GLOB))
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
            userid_idx, rt_userid_idx = get_midterm_indices(path)
            for row in read_midterm_rows(path):
                rows_seen += 1
                src = normalize_user_id(row[userid_idx])
                dst = normalize_user_id(row[rt_userid_idx])
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

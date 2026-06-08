import ast
import glob
import time

import pandas as pd


CSV_GLOB = "/project2/ll_774_951/midterm/*/*.csv"
USECOLS = ["userid", "rt_userid", "reply_userid", "qtd_userid", "mentionid"]
CHUNKSIZE = 50000


def normalize_user_id(value):
    if pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def parse_int_list_field(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()
        if text in {"", "[]", "nan", "None"}:
            return []
        try:
            parsed = ast.literal_eval(text)
            raw_items = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            text = text.strip("[]")
            raw_items = [part.strip() for part in text.split(",") if part.strip()]

    out = []
    for item in raw_items:
        user_id = normalize_user_id(item)
        if user_id is not None:
            out.append(user_id)
    return out


def iter_chunks(path):
    return pd.read_csv(
        path,
        chunksize=CHUNKSIZE,
        low_memory=False,
        on_bad_lines="skip",
        engine="c",
    )


files = sorted(glob.glob(CSV_GLOB))
print(f"Found {len(files)} files", flush=True)

total_rows = 0
unique_ids = set()
start = time.time()

for i, path in enumerate(files, 1):
    try:
        file_rows = 0
        for chunk in iter_chunks(path):
            present_cols = [c for c in USECOLS if c in chunk.columns]
            if not present_cols:
                continue
            chunk = chunk[present_cols]
            file_rows += len(chunk)

            for col in ("userid", "rt_userid", "reply_userid", "qtd_userid"):
                if col in chunk.columns:
                    vals = pd.to_numeric(chunk[col], errors="coerce").dropna().astype("int64")
                    unique_ids.update(vals.tolist())

            if "mentionid" in chunk.columns:
                for mention_ids in chunk["mentionid"]:
                    unique_ids.update(parse_int_list_field(mention_ids))

        total_rows += file_rows
    except Exception as exc:
        print(f"  skipped {path}: {exc}", flush=True)

    if i % 10 == 0 or i == len(files):
        elapsed = time.time() - start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(files) - i) / rate if rate > 0 else 0
        print(
            f"[{i:>4}/{len(files)}] "
            f"rows={total_rows:>12,}  "
            f"unique_users={len(unique_ids):>10,}  "
            f"elapsed={elapsed/60:5.1f}m  "
            f"eta={eta/60:5.1f}m",
            flush=True,
        )

print(f"\nFinal: {total_rows:,} rows, {len(unique_ids):,} unique user ids")
print(f"Took {(time.time() - start)/60:.1f} minutes")

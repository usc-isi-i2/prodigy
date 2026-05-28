import polars as pl
import glob
import sys
import time

files = sorted(glob.glob("/project2/ll_774_951/uk_ru/twitter/data/*/*.csv"))
print(f"Found {len(files)} files", flush=True)

total_rows = 0
unique_ids: set = set()
start = time.time()

for i, f in enumerate(files, 1):
    try:
        df = pl.read_csv(f, columns=["userid"], ignore_errors=True)
        total_rows += df.height
        unique_ids.update(df["userid"].unique().to_list())
    except Exception as e:
        print(f"  skipped {f}: {e}", flush=True)

    # progress every 10 files + always on the last one
    if i % 10 == 0 or i == len(files):
        elapsed = time.time() - start
        rate = i / elapsed
        eta = (len(files) - i) / rate if rate > 0 else 0
        print(
            f"[{i:>4}/{len(files)}] "
            f"rows={total_rows:>12,}  "
            f"unique={len(unique_ids):>10,}  "
            f"elapsed={elapsed/60:5.1f}m  "
            f"eta={eta/60:5.1f}m",
            flush=True,
        )

print(f"\nFinal: {total_rows:,} rows, {len(unique_ids):,} unique userids")
print(f"Took {(time.time()-start)/60:.1f} minutes")
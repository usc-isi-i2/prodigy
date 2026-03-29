#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from collections import Counter

import pandas as pd


def load_interleaved_csv(filepath):
    main_rows, sub_rows = [], []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            sub_header_raw = next(reader)
        except StopIteration:
            return pd.DataFrame()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        if sub_header_raw is not None:
            next(reader, None)

        pending_main = None
        for row in reader:
            if not row:
                continue

            if len(row) == 66:
                if pending_main is not None:
                    main_rows.append(pending_main)
                    sub_rows.append([""] * 11)
                pending_main = row
            elif len(row) == 11:
                if pending_main is not None:
                    main_rows.append(pending_main)
                    sub_rows.append(row)
                    pending_main = None
            else:
                continue

        if pending_main is not None:
            main_rows.append(pending_main)
            sub_rows.append([""] * 11)

    sub_cols = [
        "sub_extra",
        "state",
        "country",
        "rt_state",
        "rt_country",
        "qtd_state",
        "qtd_country",
        "norm_country",
        "norm_rt_country",
        "norm_qtd_country",
        "acc_age",
    ]

    df_main = pd.DataFrame(main_rows, columns=header)
    df_sub = pd.DataFrame(sub_rows, columns=sub_cols).drop(columns=["sub_extra"], errors="ignore")
    return pd.concat([df_main.reset_index(drop=True), df_sub.reset_index(drop=True)], axis=1)


def normalize_text(series):
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.isin(["", "nan", "none", "<na>"]))


def maybe_col(df, col):
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return normalize_text(df[col])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/project2/ll_774_951/uk_ru/twitter/data/*/*.csv")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sample", type=int, default=20, help="How many example blank rows to print")
    args = parser.parse_args()

    files = sorted(glob.glob(args.csv))
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern: {args.csv}")

    total_rows = 0
    total_blank_rt = 0
    total_retweet_rows = 0
    total_blank_rt_with_rtid = 0
    total_blank_rt_with_rttext = 0
    total_blank_rt_with_type_retweet = 0

    blank_by_file = Counter()
    blank_type_counter = Counter()
    blank_examples = []

    for i, filepath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(filepath)}")
        df = load_interleaved_csv(filepath)
        if df.empty:
            continue

        total_rows += len(df)

        screen_name = maybe_col(df, "screen_name")
        rt_screen = maybe_col(df, "rt_screen")
        rt_userid = maybe_col(df, "rt_userid")
        rt_text = maybe_col(df, "rt_text")
        tweet_type = maybe_col(df, "tweet_type")
        tweetid = maybe_col(df, "tweetid")

        is_retweet_like = rt_userid.notna() | rt_text.notna() | tweet_type.str.lower().eq("retweet")
        blank_rt = rt_screen.isna()
        blank_rt_retweet_like = blank_rt & is_retweet_like

        total_blank_rt += int(blank_rt.sum())
        total_retweet_rows += int(is_retweet_like.sum())
        total_blank_rt_with_rtid += int((blank_rt & rt_userid.notna()).sum())
        total_blank_rt_with_rttext += int((blank_rt & rt_text.notna()).sum())
        total_blank_rt_with_type_retweet += int((blank_rt & tweet_type.str.lower().eq("retweet")).sum())

        if blank_rt_retweet_like.any():
            blank_by_file[os.path.basename(filepath)] += int(blank_rt_retweet_like.sum())

        for value, count in tweet_type[blank_rt_retweet_like].fillna("<missing>").value_counts().items():
            blank_type_counter[str(value)] += int(count)

        if len(blank_examples) < args.sample:
            sample_df = pd.DataFrame(
                {
                    "file": os.path.basename(filepath),
                    "tweetid": tweetid,
                    "screen_name": screen_name,
                    "tweet_type": tweet_type,
                    "rt_screen": rt_screen,
                    "rt_userid": rt_userid,
                    "rt_text": rt_text,
                }
            )
            sample_df = sample_df[blank_rt_retweet_like].head(args.sample - len(blank_examples))
            blank_examples.extend(sample_df.to_dict("records"))

    print("\nSummary")
    print(f"  Total reconstructed rows: {total_rows:,}")
    print(f"  Retweet-like rows: {total_retweet_rows:,}")
    print(f"  Rows with blank rt_screen: {total_blank_rt:,}")
    print(f"  Blank rt_screen + rt_userid present: {total_blank_rt_with_rtid:,}")
    print(f"  Blank rt_screen + rt_text present: {total_blank_rt_with_rttext:,}")
    print(f"  Blank rt_screen + tweet_type=retweet: {total_blank_rt_with_type_retweet:,}")

    print("\nTop files with blank rt_screen on retweet-like rows")
    for name, count in blank_by_file.most_common(10):
        print(f"  {name}: {count:,}")

    print("\nTweet types among blank rt_screen retweet-like rows")
    for name, count in blank_type_counter.most_common(10):
        print(f"  {name}: {count:,}")

    print("\nSample rows")
    for row in blank_examples:
        rt_text_preview = row["rt_text"]
        if pd.notna(rt_text_preview):
            rt_text_preview = str(rt_text_preview).replace("\n", " ")[:120]
        print(
            f"  file={row['file']} tweetid={row['tweetid']} "
            f"screen_name={row['screen_name']} tweet_type={row['tweet_type']} "
            f"rt_userid={row['rt_userid']} rt_screen={row['rt_screen']} "
            f"rt_text={rt_text_preview}"
        )


if __name__ == "__main__":
    main()

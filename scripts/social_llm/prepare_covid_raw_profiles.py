"""Restore raw COVID Political bios while preserving canonical row order and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    canonical = pd.read_csv(args.canonical, dtype=str, keep_default_na=False)
    full = pd.read_csv(args.full, dtype=str, keep_default_na=False)
    if "raw_profile" not in full.columns:
        raise KeyError("full CSV has no raw_profile column")

    join_cols = list(canonical.columns)
    full["verified"] = full["verified"].map({"False": "0", "True": "1"}).fillna(full["verified"])
    lookup = (
        full.groupby(join_cols, dropna=False)
        .agg(match_count=("raw_profile", "size"), raw_count=("raw_profile", "nunique"), raw_profile=("raw_profile", "first"))
        .reset_index()
    )
    merged = canonical.merge(lookup, on=join_cols, how="left", validate="one_to_one")
    if not merged["match_count"].eq(1).all() or not merged["raw_count"].eq(1).all():
        raise RuntimeError("canonical rows do not map one-to-one to raw profiles")
    if merged["raw_profile"].eq("").any():
        raise RuntimeError("matched raw_profile contains empty rows")

    repaired = canonical.copy()
    repaired["profile"] = merged["raw_profile"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=False)
    repaired.to_csv(out, index=False)
    reread = pd.read_csv(out, dtype=str, keep_default_na=False)
    if not reread.equals(repaired):
        raise RuntimeError("CSV round-trip mismatch")
    report = {
        "canonical_csv": args.canonical,
        "full_csv": args.full,
        "out": str(out),
        "rows": len(repaired),
        "one_to_one_matches": int(merged["match_count"].eq(1).sum()),
        "changed_profiles": int((canonical["profile"] != repaired["profile"]).sum()),
        "empty_raw_profiles": int(repaired["profile"].eq("").sum()),
        "non_profile_columns_identical": bool(repaired.drop(columns="profile").equals(canonical.drop(columns="profile"))),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

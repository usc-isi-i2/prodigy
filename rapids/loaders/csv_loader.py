"""CSV loaders for midterm (standard) and ukr_rus_twitter (interleaved) datasets."""
import csv

import pandas as pd

from rapids.loaders.base import RecordLoader


def load_standard_csv(filepath: str) -> pd.DataFrame:
    """Load a plain CSV file, replacing empty strings with NA."""
    try:
        df = pd.read_csv(filepath, low_memory=False, on_bad_lines="skip")
        df.replace("", pd.NA, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_interleaved_csv(filepath: str) -> pd.DataFrame:
    """Load the ukr_rus_twitter interleaved format.

    Each tweet occupies two consecutive rows: a 66-column main row followed by
    an optional 11-column sub-row containing geo enrichment fields.
    """
    main_rows, sub_rows = [], []

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            sub_header_raw = next(reader)
        except StopIteration:
            return pd.DataFrame()

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        if sub_header_raw is not None:
            next(reader)

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
            # skip rows of unexpected width

        if pending_main is not None:
            main_rows.append(pending_main)
            sub_rows.append([""] * 11)

    sub_cols = [
        "sub_extra", "state", "country", "rt_state", "rt_country",
        "qtd_state", "qtd_country", "norm_country", "norm_rt_country",
        "norm_qtd_country", "acc_age",
    ]
    df_main = pd.DataFrame(main_rows, columns=header)
    df_sub = (
        pd.DataFrame(sub_rows, columns=sub_cols)
        .drop(columns=["sub_extra"], errors="ignore")
    )
    df = pd.concat([df_main.reset_index(drop=True), df_sub.reset_index(drop=True)], axis=1)
    df.replace("", pd.NA, inplace=True)
    return df


def load_ukr_rus_file(fpath: str) -> pd.DataFrame:
    """Try interleaved format first; fall back to standard CSV on failure."""
    try:
        df = load_interleaved_csv(fpath)
        if not df.empty:
            return df
    except Exception:
        pass
    return load_standard_csv(fpath)


class StandardCsvLoader(RecordLoader):
    """``RecordLoader`` for plain CSV files (e.g. midterm dataset)."""

    def load_dataframe(self, path: str) -> pd.DataFrame:
        return load_standard_csv(path)


class UkrRusLoader(RecordLoader):
    """``RecordLoader`` for ukr_rus_twitter interleaved CSV files."""

    def load_dataframe(self, path: str) -> pd.DataFrame:
        return load_ukr_rus_file(path)

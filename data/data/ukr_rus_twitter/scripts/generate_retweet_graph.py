import argparse
import ast
import csv
import glob
import json
import os
import re
from urllib.parse import urlparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


DEFAULT_CSV = "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv"
DEFAULT_OUT = "data/data/ukr_rus_twitter/graphs/retweet_graph.pt"
DEFAULT_HISTORY_FRACTION = 0.3
LABEL_NAMES = ["left", "right"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

OUTLET_SCORES = {
    "abcnews.go.com": 2.0,
    "bbc.com": 3.0,
    "breitbart.com": 5.0,
    "bostonglobe.com": 2.0,
    "businessinsider.com": 3.0,
    "buzzfeednews.com": 1.0,
    "cbsnews.com": 2.0,
    "chicagotribune.com": 3.0,
    "cnbc.com": 3.0,
    "cnn.com": 2.0,
    "dailycaller.com": 5.0,
    "dailymail.co.uk": 5.0,
    "foxnews.com": 4.0,
    "huffpost.com": 1.0,
    "infowars.com": 5.0,
    "latimes.com": 2.0,
    "msnbc.om": 1.0,
    "nbcnews.com": 2.0,
    "nytimes.com": 2.0,
    "npr.org": 3.0,
    "oann.com": 4.0,
    "pbs.org": 3.0,
    "reuters.com": 3.0,
    "theguardian.com": 2.0,
    "usatoday.com": 3.0,
    "yahoo.com": 2.0,
    "vice.com": 1.0,
    "washingtonpost.com": 2.0,
    "wsj.com": 3.0,
    # additional
    "thehill.com": 2.7,
    "rt.com": 3.7,
    "rawstory.com": 1.7,
    "news.sky.com": 2.3,
    "independent.co.uk": 2.3,
    "dailykos.com": 1.7,
}

HASHTAG_TO_LABEL = {'blm': 0, 'resist': 0, 'fbpe': 0, 'blacklivesmatter': 0, 'fbr': 0, 'maga': 1, 'theresistance': 0, 'voteblue': 0, 'resistance': 0, 'bidenharris': 0, 'johnsonout': 0, 'lgbtq': 0, 'bidenharris2020': 0, '2a': 1, 'fbpa': 0, 'resister': 0, 'fbppr': 0, 'bluewave': 0, 'gtto': 0, 'freepalestine': 0, 'rejoineu': 0, 'voteblue2022': 0, 'kag': 1, 'wearamask': 0, 'getvaccinated': 0, 'humanrights': 0, 'votingrights': 0, 'science': 0, 'goodtrouble': 0, 'strongertogether': 0, 'stillwithher': 0, 'climatecrisis': 0, 'metoo': 0, 'demvoice1': 0, 'biden': 0, 'climatechange': 0, 'justicematters': 0, 'americafirst': 1, 'nevertrump': 0, 'khive': 0, 'democrat': 0, 'vaccinated': 0, 'buildbackbetter': 0, 'stopasianhate': 0, 'prochoice': 0, 'drcole': 1, 'bds': 0, 'votebluenomatterwho': 0, 'teampelosi': 0, 'handmarkedpaperballots': 1, 'reconquête': 1, 'biden2020': 0, 'patriot': 1, 'equality': 0, 'prolife': 1, 'antifa': 0, 'fjb': 1, 'lgbt': 0, 'nra': 1, 'climateaction': 0, 'unionpopulaire': 0, 'zemmour2022': 1, 'trump': 1, 'demcast': 0, 'm4a': 0, 'vaxxed': 0, 'democracy': 0, 'indivisible': 0, 'teamjustice': 0, 'noafd': 0, 'alllivesmatter': 1}


DOMAIN_TO_CANONICAL = {
    "bit.ly": None,
    "dlvr.it": None,
    "trib.al": None,
    "ift.tt": None,
    "twibbon.com": None,
    "t.me": None,
    "apple.news": None,
    "ow.ly": None,
    "buff.ly": None,
    "tinyurl.com": None,
    "news.google.com": None,
    "lnr.app": None,
    "fiverr.com": None,
    "twitter.com": "twitter.com",
    "reut.rs": "reuters.com",
    "youtu.be": "youtube.com",
    "theguardian.com": "theguardian.com",
    "youtube.com": "youtube.com",
    "nytimes.com": "nytimes.com",
    "reuters.com": "reuters.com",
    "mfa.gov.tr": "mfa.gov.tr",
    "washingtonpost.com": "washingtonpost.com",
    "cnn.com": "cnn.com",
    "businessinsider.com": "businessinsider.com",
    "rt.com": "rt.com",
    "hill.cm": "thehill.com",
    "bbc.in": "bbc.com",
    "foxnews.com": "foxnews.com",
    "facebook.com": "facebook.com",
    "rawstory.com": "rawstory.com",
    "bbc.co.uk": "bbc.com",
    "news.sky.com": "news.sky.com",
    "melenchon.fr": "melenchon.fr",
    "a.msn.com": "msn.com",
    "npr.org": "npr.org",
    "politico.com": "politico.com",
    "en.wikipedia.org": "wikipedia.org",
    "cnb.cx": "cnbc.com",
    "timesofindia.indiatimes.com": "timesofindia.indiatimes.com",
    "dailykos.com": "dailykos.com",
    "independent.co.uk": "independent.co.uk",
    "bfmtv.com": "bfmtv.com",
    "msn.com": "msn.com",
    "jp.reuters.com": "reuters.com",
    "axios.com": "axios.com",
    "babylonbee.com": "babylonbee.com",
    "news.yahoo.com": "yahoo.com",
    "gelecektenhaber.com": "gelecektenhaber.com",
    "instagram.com": "instagram.com",
    "google.com": "google.com",
    "zerohedge.com": "zerohedge.com",
    "tagesschau.de": "tagesschau.de",
    "thehill.com": "thehill.com",
    "bloomberg.com": "bloomberg.com",
    "lemonde.fr": "lemonde.fr",
    "ndtv.com": "ndtv.com",
    "forbes.com": "forbes.com",
    "ti.me": "time.com",
    "aninews.in": "aninews.in",
    "theatlantic.com": "theatlantic.com",
    "spiegel.de": "spiegel.de",
    "on.ft.com": "ft.com",
    "en.kremlin.ru": "kremlin.ru",
    "cnn.it": "cnn.com",
    "whitehouse.gov": "whitehouse.gov",
    "insiderpaper.com": "insiderpaper.com",
    "abc.net.au": "abc.net.au",
    "francetvinfo.fr": "francetvinfo.fr",
    "edition.cnn.com": "cnn.com",
    "on.rt.com": "rt.com",
    "aljazeera.com": "aljazeera.com",
    "mobile.twitter.com": "twitter.com",
    "parafesor.net": "parafesor.net",
    "who.int": "who.int",
    "world.bistosh.com": "bistosh.com",
    "es.rt.com": "rt.com",
    "coinkurry.com": "coinkurry.com",
    "presidentti.fi": "presidentti.fi",
    "nos.nl": "nos.nl",
    "sputniknews.com": "sputniknews.com",
    "mol.im": "dailymail.co.uk",
    "vilaweb.cat": "vilaweb.cat",
    "bild.de": "bild.de",
    "washingtontimes.com": "washingtontimes.com",
    "scmp.com": "scmp.com",
    "wapo.st": "washingtonpost.com",
    "aajtak.in": "aajtak.in",
    "nbcnews.com": "nbcnews.com",
    "tf1info.fr": "tf1info.fr",
    "rumble.com": "rumble.com",
    "maisinwin.blogspot.com": "maisinwin.blogspot.com",
    "welt.de": "welt.de",
    "nypost.com": "nypost.com",
    "ft.com": "ft.com",
    "liveuamap.com": "liveuamap.com",
    "moneycontrol.com": "moneycontrol.com",
    "kremlin.ru": "kremlin.ru",
}

HANDLE_SCORES = {
    'abc': 2,
    'bbcworld': 3,
    'breitbartnews': 5,
    'bostonglobe': 2,
    'businessinsider': 3,
    'buzzfeednews': 1,
    'cbsnews': 2,
    'chicagotribune': 3,
    'cnbc': 3,
    'cnn': 2,
    'dailycaller': 5,
    'dailymail': 5,
    'foxnews': 4,
    'huffpost': 1,
    'infowars': 5,
    'latimes': 2,
    'msnbc': 1,
    'nbcnews': 2,
    'nytimes': 2,
    'npr': 3,
    'oann': 4,
    'pbs': 3,
    'reuters': 3,
    'guardian': 2,
    'usatoday': 3,
    'yahoonews': 2,
    'vice': 1,
    'washingtonpost': 2,
    'wsj': 3,
    # additional
    'thehill': 2.7,
    'rt_com': 3.7,
    'rawstory': 1.7,
    'skynews': 2.3,
    'independent': 2.3,
    'dailykos': 1.7,
}

URL_TO_LABEL = {
    domain: ("left" if score < 3 else "right")
    for domain, score in OUTLET_SCORES.items()
    if score != 3
}

NODE_FEATURE_NAMES = [
    "subscriber_count",
    "verified",
    "avg_favorites",
    "avg_comments",
    "avg_score",
    "avg_n_hashtags",
    "avg_n_mentions",
    "avg_has_media",
    "post_count",
    "in_degree",
    "out_degree",
]

EDGE_FEATURE_NAMES = [
    "first_retweet_time",
    "n_retweets",
    "avg_rt_fav",
    "avg_rt_reply",
]

USECOLS = {
    "screen_name",
    "userid",
    "rt_userid",
    "rt_screen",
    "date",
    "followers_count",
    "verified",
    "statuses_count",
    "rt_fav_count",
    "rt_reply_count",
    "sent_vader",
    "hashtag",
    "mentionsn",
    "media_urls",
    "description",
    "urls_list",
    "rt_urls_list",
    "rt_hashtag",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Build unified ukr_rus_twitter retweet graph with optional embeddings and temporal views."
    )
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--embeddings", default="", help="Optional user_embeddings_*.pt with user_ids + meanpool (handles fallback supported)")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=0, help="Trim the final graph to exactly this many nodes when possible (0 = no limit)")
    p.add_argument("--strict_dates", action="store_true")
    p.add_argument("--history_fraction", type=float, default=DEFAULT_HISTORY_FRACTION)
    p.add_argument(
        "--future_target_mode",
        choices=["new_only", "all_future"],
        default="new_only",
        help="How to define temporal LP targets from the future slice.",
    )
    p.add_argument("--no_temporal_views", action="store_true")
    p.add_argument(
        "--keep-isolates",
        dest="keep_isolates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep nodes with zero in-degree and zero out-degree in the saved graph.",
    )
    p.add_argument(
        "--pseudo-political-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate binary pseudo political labels from linked news domains.",
    )
    p.add_argument(
        "--pseudo-label-margin",
        type=int,
        default=2,
        help="Minimum absolute left-vs-right evidence difference required to assign a pseudo label.",
    )
    return p.parse_args()


def load_interleaved_csv(filepath: str) -> pd.DataFrame:
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


def normalize_handle(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.mask(normalized.isin(["", "nan", "none", "<na>"]))


def count_list_like(val) -> int:
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s in {"", "[]", "nan", "None"}:
        return 0
    s = s.strip("[]").replace("'", "").replace('"', "")
    if not s:
        return 0
    return len([x for x in s.split(",") if x.strip()])


def _fast_list_lens(series: pd.Series) -> np.ndarray:
    return np.fromiter(
        (count_list_like(x) for x in series),
        dtype=np.int32,
        count=len(series),
    )


def parse_serialized_list(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if s in {"", "[]", "nan", "None", "<NA>"}:
        return []
    try:
        parsed = ast.literal_eval(s)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_domain(domain: str) -> str:
    domain = (domain or "").strip().lower()
    if not domain:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def extract_domains_from_url_field(val) -> List[str]:
    domains: List[str] = []
    for item in parse_serialized_list(val):
        expanded = ""
        if isinstance(item, dict):
            expanded = str(item.get("expanded_url") or item.get("url") or "").strip()
        else:
            expanded = str(item).strip()
        if not expanded:
            continue
        try:
            netloc = urlparse(expanded).netloc
        except Exception:
            netloc = ""
        domain = normalize_domain(netloc)
        if domain:
            domains.append(domain)
    return domains


def canonicalize_domain(domain: str):
    domain = normalize_domain(domain)
    if not domain:
        return None
    if domain in DOMAIN_TO_CANONICAL:
        return DOMAIN_TO_CANONICAL[domain]
    if domain in URL_TO_LABEL:
        return domain
    parts = domain.split(".")
    for i in range(1, len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in DOMAIN_TO_CANONICAL:
            return DOMAIN_TO_CANONICAL[candidate]
        if candidate in URL_TO_LABEL:
            return candidate
    return None


def score_url_field_value(val) -> Dict[str, int]:
    counts = {label: 0 for label in LABEL_NAMES}
    for domain in extract_domains_from_url_field(val):
        canonical = canonicalize_domain(domain)
        if canonical is None:
            continue
        label = URL_TO_LABEL.get(canonical)
        if label is None:
            continue
        counts[label] += 1
    return counts


def map_url_scores(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    keys = series.astype("string").fillna("")
    unique_vals = pd.unique(keys)

    left_map: Dict[str, int] = {}
    right_map: Dict[str, int] = {}
    for val in unique_vals:
        counts = score_url_field_value(val)
        left_map[val] = counts["left"]
        right_map[val] = counts["right"]

    left_series = keys.map(left_map).fillna(0).astype(np.int32)
    right_series = keys.map(right_map).fillna(0).astype(np.int32)
    return left_series, right_series


def extract_hashtags_from_field(val) -> List[str]:
    tags: List[str] = []
    for item in parse_serialized_list(val):
        if isinstance(item, dict):
            tag = str(item.get("text") or item.get("tag") or "").strip().lower()
        else:
            tag = str(item).strip().lower()
        if not tag:
            continue
        if tag.startswith("#"):
            tag = tag[1:]
        if tag:
            tags.append(tag)
    return tags


def extract_hashtags_from_text(text) -> List[str]:
    s = str(text or "").lower()
    return [match[1:] for match in re.findall(r"#\w+", s)]


def score_hashtag_iter(tags: List[str]) -> Dict[str, int]:
    counts = {label: 0 for label in LABEL_NAMES}
    for tag in tags:
        val = HASHTAG_TO_LABEL.get(tag)
        if val is None:
            continue
        label = LABEL_NAMES[val]
        counts[label] += 1
    return counts


def map_hashtag_field_scores(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    keys = series.astype("string").fillna("")
    unique_vals = pd.unique(keys)

    left_map: Dict[str, int] = {}
    right_map: Dict[str, int] = {}
    for val in unique_vals:
        counts = score_hashtag_iter(extract_hashtags_from_field(val))
        left_map[val] = counts["left"]
        right_map[val] = counts["right"]

    left_series = keys.map(left_map).fillna(0).astype(np.int32)
    right_series = keys.map(right_map).fillna(0).astype(np.int32)
    return left_series, right_series


def build_pseudo_political_labels(
    raw: pd.DataFrame,
    handles: List[str],  # screen_name aligned with user_ids
    min_margin: int = 2,
) -> Tuple[torch.Tensor, List[str], int]:
    # Signals: own-tweet URLs, retweet target handles, description hashtags — all keyed by screen_name
    needed_cols = [c for c in ("screen_name", "rt_screen", "description", "urls_list") if c in raw.columns]
    user_df = raw[needed_cols].copy()
    user_df["screen_name"] = normalize_handle(user_df["screen_name"])
    user_df = user_df[user_df["screen_name"].notna()].copy()
    if user_df.empty:
        y = torch.full((len(handles),), -1, dtype=torch.long)
        return y, list(LABEL_NAMES), 0

    # Signal 1: URLs in own tweets (urls_list only, not rt_urls_list)
    url_left = pd.Series(0, index=user_df.index, dtype=np.int32)
    url_right = pd.Series(0, index=user_df.index, dtype=np.int32)
    if "urls_list" in user_df.columns:
        url_left, url_right = map_url_scores(user_df["urls_list"])

    url_agg = (
        pd.DataFrame(
            {
                "screen_name": user_df["screen_name"],
                "left_url": url_left,
                "right_url": url_right,
            }
        )
        .groupby("screen_name", sort=False)[["left_url", "right_url"]]
        .sum()
    )

    # Signal 2: Retweet target handles (rt_screen)
    rt_agg = pd.DataFrame(index=url_agg.index, data={"left_rt": 0, "right_rt": 0})
    if "rt_screen" in user_df.columns:
        rt_score = user_df["rt_screen"].str.lower().map(HANDLE_SCORES)
        rt_df = pd.DataFrame({
            "screen_name": user_df["screen_name"],
            "left_rt": (rt_score.lt(3) & rt_score.notna()).astype(np.int32).values,
            "right_rt": (rt_score.gt(3) & rt_score.notna()).astype(np.int32).values,
        })
        rt_agg = rt_df.groupby("screen_name", sort=False)[["left_rt", "right_rt"]].sum()

    # Signal 3: Hashtags in profile description
    desc_agg = pd.DataFrame(index=url_agg.index, data={"left_hashtag": 0, "right_hashtag": 0})
    if "description" in user_df.columns:
        desc_df = user_df[["screen_name", "description"]].copy()
        desc_df["description"] = desc_df["description"].astype("string").fillna("").str.strip().str.lower()
        desc_df = desc_df[desc_df["description"] != ""].drop_duplicates(subset=["screen_name", "description"])
        if not desc_df.empty:
            desc_scores = desc_df["description"].apply(lambda text: score_hashtag_iter(extract_hashtags_from_text(text)))
            desc_df["left_hashtag"] = desc_scores.map(lambda d: d["left"]).astype(np.int32)
            desc_df["right_hashtag"] = desc_scores.map(lambda d: d["right"]).astype(np.int32)
            desc_agg = desc_df.groupby("screen_name", sort=False)[["left_hashtag", "right_hashtag"]].sum()

    label_agg = url_agg.join(rt_agg, how="outer").join(desc_agg, how="outer").fillna(0)
    left_totals = label_agg["left_url"] + label_agg["left_rt"] + label_agg["left_hashtag"]
    right_totals = label_agg["right_url"] + label_agg["right_rt"] + label_agg["right_hashtag"]

    handle_index = pd.Index(handles)
    left_votes = handle_index.map(left_totals).fillna(0).astype(int)
    right_votes = handle_index.map(right_totals).fillna(0).astype(int)

    has_enough = (right_votes >= min_margin) | (left_votes >= min_margin)
    not_mixed = ~((right_votes > 0) & (left_votes > 0))
    valid = has_enough & not_mixed

    y_np = np.full(len(handles), -1, dtype=np.int64)
    y_np[valid & (right_votes > 0)] = LABEL_TO_ID["right"]
    y_np[valid & (right_votes == 0)] = LABEL_TO_ID["left"]
    labeled = int(valid.sum())
    return torch.from_numpy(y_np), list(LABEL_NAMES), labeled


def load_raw_rows(csv_glob: str, max_files: int) -> pd.DataFrame:
    files = sorted(glob.glob(csv_glob))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {csv_glob}")

    chunks: List[pd.DataFrame] = []
    print(f"Found {len(files)} files")
    for i, fpath in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Loading {os.path.basename(fpath)}", flush=True)
        try:
            dfi = load_interleaved_csv(fpath)
            if dfi.empty:
                print("  [SKIP] empty file", flush=True)
                continue
            cols = [c for c in dfi.columns if c in USECOLS]
            if not cols:
                print("  [SKIP] no required columns", flush=True)
                continue
            chunks.append(dfi[cols].copy())
        except Exception as exc:
            print(f"  [WARN] failed {os.path.basename(fpath)}: {exc}", flush=True)

        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files", flush=True)

    if not chunks:
        raise RuntimeError("No valid rows parsed")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded rows: {len(df):,}")
    return df


def prepare_retweet_rows(df: pd.DataFrame, strict_dates: bool) -> pd.DataFrame:
    df = df.copy()
    for col in ["userid", "rt_userid", "followers_count", "verified", "statuses_count", "rt_fav_count", "rt_reply_count", "sent_vader"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "screen_name" in df.columns:
        df["screen_name"] = normalize_handle(df["screen_name"])
    if "rt_screen" in df.columns:
        df["rt_screen"] = normalize_handle(df["rt_screen"])

    df["date"] = df.get("date", "").astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(
        df["date"],
        format="%a %b %d %H:%M:%S +0000 %Y",
        utc=True,
        errors="coerce",
    )

    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        n_bad = int(bad_ts.sum())
        examples = df.loc[bad_ts, "date"].drop_duplicates().head(5).tolist()
        if strict_dates:
            raise ValueError(f"Invalid timestamp rows: {n_bad:,}, examples={examples}")
        print(f"Dropping invalid timestamp rows: {n_bad:,}")
        df = df.loc[~bad_ts].copy()

    rt = df.dropna(subset=["userid", "rt_userid"]).copy()
    rt["userid"] = rt["userid"].astype(np.int64)
    rt["rt_userid"] = rt["rt_userid"].astype(np.int64)
    rt = rt[rt["userid"] != rt["rt_userid"]].copy()
    if rt.empty:
        raise RuntimeError("No valid retweet rows after cleaning")
    return rt


def trim_rt_to_max_nodes(rt: pd.DataFrame, max_nodes: int) -> pd.DataFrame:
    if max_nodes <= 0:
        return rt

    total_unique_nodes = len(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist()))
    if total_unique_nodes < max_nodes:
        raise ValueError(
            f"Requested max_nodes={max_nodes:,}, but only {total_unique_nodes:,} unique retweet nodes exist "
            "after cleaning."
        )

    src = rt["userid"].to_numpy(dtype=np.int64, copy=False)
    dst = rt["rt_userid"].to_numpy(dtype=np.int64, copy=False)
    keep_rows = np.zeros(len(rt), dtype=bool)
    keep_nodes = set()

    for i, (u, v) in enumerate(zip(src, dst)):
        add = 0
        if u not in keep_nodes:
            add += 1
        if v not in keep_nodes:
            add += 1
        if len(keep_nodes) + add <= max_nodes:
            keep_rows[i] = True
            keep_nodes.add(int(u))
            keep_nodes.add(int(v))
        elif len(keep_nodes) >= max_nodes and u in keep_nodes and v in keep_nodes:
            keep_rows[i] = True
        if len(keep_nodes) == max_nodes:
            continue

    rt2 = rt.loc[keep_rows].copy()
    actual_nodes = sorted(set(rt2["userid"].tolist()) | set(rt2["rt_userid"].tolist()))
    print(
        f"Applied exact max_nodes={max_nodes:,}: kept rows={len(rt2):,} "
        f"nodes={len(actual_nodes):,}",
        flush=True,
    )
    if len(actual_nodes) != max_nodes:
        raise RuntimeError(
            f"Exact max_nodes trim failed: requested {max_nodes:,}, got {len(actual_nodes):,} nodes"
        )
    return rt2


def build_user_index(rt: pd.DataFrame) -> Tuple[List[int], Dict[int, int]]:
    user_ids = sorted(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist()))
    return user_ids, {int(uid): i for i, uid in enumerate(user_ids)}


def build_user_metadata(rt: pd.DataFrame, user_ids: List[int]) -> List[str]:
    """Build a list of screen_name handles aligned with user_ids, for label computation."""
    frames = []
    if "screen_name" in rt.columns:
        left = rt[["userid", "screen_name", "timestamp"]].rename(
            columns={"userid": "user_id", "screen_name": "handle"}
        )
        frames.append(left)
    if "rt_screen" in rt.columns:
        right = rt[["rt_userid", "rt_screen", "timestamp"]].rename(
            columns={"rt_userid": "user_id", "rt_screen": "handle"}
        )
        frames.append(right)
    if not frames:
        return [None] * len(user_ids)
    meta = pd.concat(frames, ignore_index=True, copy=False)
    meta = meta.dropna(subset=["user_id", "handle"])
    if meta.empty:
        return [None] * len(user_ids)
    latest_idx = meta.groupby("user_id")["timestamp"].idxmax()
    latest = meta.loc[latest_idx, ["user_id", "handle"]]
    handle_map = dict(zip(latest["user_id"].to_numpy(), latest["handle"].to_numpy()))
    return [handle_map.get(int(uid)) for uid in user_ids]


def aggregate_edge_features(rt: pd.DataFrame) -> pd.DataFrame:
    edge_grp = rt.groupby(["userid", "rt_userid"], as_index=False).agg(
        first_ts=("timestamp", "min"),
        n_retweets=("timestamp", "size"),
        avg_rt_fav=("rt_fav_count", "mean"),
        avg_rt_reply=("rt_reply_count", "mean"),
    )
    min_ns = int(edge_grp["first_ts"].min().value)
    edge_grp["first_retweet_time"] = (edge_grp["first_ts"].astype("int64") - min_ns) / 3_600_000_000_000.0
    edge_grp["n_retweets"] = np.log1p(edge_grp["n_retweets"].astype(float))
    edge_grp["avg_rt_fav"] = np.log1p(pd.to_numeric(edge_grp["avg_rt_fav"], errors="coerce").fillna(0).clip(lower=0))
    edge_grp["avg_rt_reply"] = np.log1p(pd.to_numeric(edge_grp["avg_rt_reply"], errors="coerce").fillna(0).clip(lower=0))
    return edge_grp


def to_edge_tensors(edge_df: pd.DataFrame, u2i: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    tmp = edge_df.copy()
    tmp["src"] = tmp["userid"].map(u2i)
    tmp["dst"] = tmp["rt_userid"].map(u2i)
    tmp = tmp.dropna(subset=["src", "dst"]).copy()
    tmp[["src", "dst"]] = tmp[["src", "dst"]].astype(int)
    edge_index = torch.tensor(tmp[["src", "dst"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(tmp[EDGE_FEATURE_NAMES].fillna(0).values.astype(np.float32), dtype=torch.float)
    return edge_index, edge_attr


def build_node_features(rt: pd.DataFrame, u2i: Dict[int, int], edge_df: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
    base = rt.copy()
    for col in ["followers_count", "statuses_count", "sent_vader", "rt_fav_count", "rt_reply_count"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    base["verified"] = base.get("verified", 0).map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(float)
    base["n_hashtags"] = _fast_list_lens(base["hashtag"]) if "hashtag" in base.columns else 0
    base["n_mentions"] = _fast_list_lens(base["mentionsn"]) if "mentionsn" in base.columns else 0
    if "media_urls" in base.columns:
        base["has_media"] = (_fast_list_lens(base["media_urls"]) > 0).astype(np.int8)
    else:
        base["has_media"] = 0

    node_agg = base.groupby("userid", as_index=False).agg(
        subscriber_count=("followers_count", "max"),
        verified=("verified", "max"),
        avg_favorites=("rt_fav_count", "mean"),
        avg_comments=("rt_reply_count", "mean"),
        avg_score=("sent_vader", "mean"),
        avg_n_hashtags=("n_hashtags", "mean"),
        avg_n_mentions=("n_mentions", "mean"),
        avg_has_media=("has_media", "mean"),
        post_count=("statuses_count", "max"),
    )

    for col in ["subscriber_count", "avg_favorites", "avg_comments", "post_count"]:
        node_agg[col] = np.log1p(pd.to_numeric(node_agg[col], errors="coerce").fillna(0).clip(lower=0))

    in_deg = edge_df.groupby("rt_userid").size().rename("in_degree")
    out_deg = edge_df.groupby("userid").size().rename("out_degree")
    node_agg = node_agg.merge(out_deg, left_on="userid", right_index=True, how="left")
    node_agg = node_agg.merge(in_deg, left_on="userid", right_index=True, how="left")
    node_agg["in_degree"] = np.log1p(node_agg["in_degree"].fillna(0))
    node_agg["out_degree"] = np.log1p(node_agg["out_degree"].fillna(0))

    n_nodes = len(u2i)
    x_np = np.zeros((n_nodes, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    node_agg["node_idx"] = node_agg["userid"].map(u2i)
    node_agg = node_agg.dropna(subset=["node_idx"])
    rows = node_agg["node_idx"].astype(int).values
    x_np[rows] = node_agg[NODE_FEATURE_NAMES].fillna(0).values.astype(np.float32)

    nonzero = np.any(x_np != 0, axis=1)
    if nonzero.any():
        scaler = StandardScaler()
        x_np[nonzero] = scaler.fit_transform(x_np[nonzero]).astype(np.float32)

    return torch.tensor(x_np, dtype=torch.float), list(NODE_FEATURE_NAMES)


def maybe_attach_embeddings(
    x: torch.Tensor,
    feature_names: List[str],
    user_ids: List[int],
    handles: List[str],
    embeddings_path: str,
    embedding_pool: str,
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    if not embeddings_path:
        return x, feature_names, {"matched_users": 0, "embedding_dim": 0}

    emb = torch.load(embeddings_path, map_location="cpu")
    emb_mat = emb.get(embedding_pool)
    if emb_mat is None:
        raise KeyError(f"Embeddings file must contain '{embedding_pool}'")

    emb_dim = int(emb_mat.shape[1])
    extra = torch.zeros((len(user_ids), emb_dim), dtype=torch.float)
    matched = 0

    emb_user_ids = emb.get("user_ids")
    if emb_user_ids is not None:
        emb_ids = np.asarray(emb_user_ids).astype(np.int64)
        order = np.argsort(emb_ids)
        sorted_ids = emb_ids[order]
        query = np.asarray(user_ids, dtype=np.int64)
        pos = np.searchsorted(sorted_ids, query)
        pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
        hit = (pos < len(sorted_ids)) & (sorted_ids[pos_clipped] == query)
        tgt_rows = np.where(hit)[0]
        src_rows = order[pos[hit]]
        if len(tgt_rows):
            extra[torch.from_numpy(tgt_rows)] = emb_mat[torch.from_numpy(src_rows)].float()
        matched = int(hit.sum())
    else:
        emb_handles = emb.get("handles")
        if emb_handles is None:
            raise KeyError(f"Embeddings file must contain either 'user_ids' or 'handles' plus '{embedding_pool}'")
        emb_handle_series = pd.Series(
            np.arange(len(emb_handles), dtype=np.int64),
            index=[str(h).strip().lower() for h in emb_handles],
        )
        emb_handle_series = emb_handle_series[~emb_handle_series.index.duplicated(keep="first")]
        mapped = pd.Series(handles).map(emb_handle_series)
        hit_mask = mapped.notna().to_numpy()
        tgt_rows = np.where(hit_mask)[0]
        src_rows = mapped.dropna().to_numpy(dtype=np.int64)
        if len(tgt_rows):
            extra[torch.from_numpy(tgt_rows)] = emb_mat[torch.from_numpy(src_rows)].float()
        matched = int(hit_mask.sum())

    x2 = torch.cat([x, extra], dim=1)
    names2 = feature_names + [f"emb_{k}" for k in range(emb_dim)]
    return x2, names2, {"matched_users": matched, "embedding_dim": emb_dim}


def drop_isolates_from_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    user_ids: List[int],
    handles: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[str], Dict[int, int], int]:
    n = len(user_ids)
    degrees_out = torch.bincount(edge_index[0], minlength=n)
    degrees_in = torch.bincount(edge_index[1], minlength=n)
    keep_mask = (degrees_out > 0) | (degrees_in > 0)
    isolated = int((~keep_mask).sum().item())
    if isolated == 0:
        return x, edge_index, edge_attr, user_ids, handles, {int(uid): i for i, uid in enumerate(user_ids)}, 0

    kept_nodes = int(keep_mask.sum().item())
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[keep_mask] = torch.arange(kept_nodes, dtype=torch.long)
    x = x[keep_mask]
    edge_index = remap[edge_index]
    if edge_attr is not None:
        edge_attr = edge_attr.clone()
    keep_list = keep_mask.tolist()
    user_ids = [uid for uid, keep in zip(user_ids, keep_list) if keep]
    handles = [h for h, keep in zip(handles, keep_list) if keep]
    u2i = {int(uid): i for i, uid in enumerate(user_ids)}
    return x, edge_index, edge_attr, user_ids, handles, u2i, isolated


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    raw = load_raw_rows(args.csv, args.max_files)
    rt = prepare_retweet_rows(raw, args.strict_dates)
    rt = trim_rt_to_max_nodes(rt, args.max_nodes)

    user_ids, u2i = build_user_index(rt)
    handles = build_user_metadata(rt, user_ids)
    print(f"Nodes: {len(user_ids):,}")

    edge_all_df = aggregate_edge_features(rt)
    edge_index, edge_attr = to_edge_tensors(edge_all_df, u2i)
    print(f"Directed edges: {edge_index.shape[1]:,}")

    x, feature_names = build_node_features(rt, u2i, edge_all_df)
    print(f"Node features: {x.shape[1]} ({', '.join(feature_names)})")

    x, feature_names, emb_stats = maybe_attach_embeddings(
        x, feature_names, user_ids, handles, args.embeddings, args.embedding_pool
    )
    print(f"After attaching embeddings: {x.shape[1]} features, matched users: {emb_stats['matched_users']:,}")

    isolated_before_drop = int(((torch.bincount(edge_index[0], minlength=len(user_ids)) == 0) & (torch.bincount(edge_index[1], minlength=len(user_ids)) == 0)).sum().item())
    if not args.keep_isolates:
        x, edge_index, edge_attr, user_ids, handles, u2i, isolated_dropped = drop_isolates_from_graph(
            x, edge_index, edge_attr, user_ids, handles
        )
        print(f"Dropped isolated nodes: {isolated_dropped:,}")
    else:
        isolated_dropped = 0

    if args.pseudo_political_labels:
        y, label_names, labeled_nodes = build_pseudo_political_labels(
            raw,
            handles,
            min_margin=args.pseudo_label_margin,
        )
        print(f"Generated pseudo political labels: {labeled_nodes:,} / {len(user_ids):,} labeled")
    else:
        y = torch.full((len(user_ids),), -1, dtype=torch.long)
        label_names = []
        labeled_nodes = 0

    graph_obj = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_FEATURE_NAMES,
        "user_ids": user_ids,
        "u2i": u2i,
        "feature_names": feature_names,
        "y": y,
        "label_names": label_names,
    }

    temporal_stats = {}
    if not args.no_temporal_views:
        rt_sorted = rt.sort_values("timestamp").reset_index(drop=True)
        cutoff_idx = int(len(rt_sorted) * args.history_fraction)
        cutoff_idx = max(1, min(len(rt_sorted) - 1, cutoff_idx))

        hist_rt = rt_sorted.iloc[:cutoff_idx].copy()
        fut_rt = rt_sorted.iloc[cutoff_idx:].copy()

        hist_edges_df = aggregate_edge_features(hist_rt)
        fut_edges_df = aggregate_edge_features(fut_rt)

        hist_edge_index, hist_edge_attr = to_edge_tensors(hist_edges_df, u2i)

        hist_pairs = set(zip(hist_edges_df["userid"], hist_edges_df["rt_userid"]))
        fut_pairs = set(zip(fut_edges_df["userid"], fut_edges_df["rt_userid"]))
        if args.future_target_mode == "new_only":
            target_pairs = fut_pairs - hist_pairs
        else:
            target_pairs = fut_pairs

        if target_pairs:
            target_df = pd.DataFrame(sorted(list(target_pairs)), columns=["userid", "rt_userid"])
            target_df["src"] = target_df["userid"].map(u2i)
            target_df["dst"] = target_df["rt_userid"].map(u2i)
            target_df = target_df.dropna(subset=["src", "dst"])
            target_new_edge_index = torch.tensor(target_df[["src", "dst"]].astype(int).values.T, dtype=torch.long)
        else:
            target_new_edge_index = torch.zeros((2, 0), dtype=torch.long)

        graph_obj["edge_index_views"] = {
            "retweet_all": edge_index,
            "temporal_history": hist_edge_index,
        }
        graph_obj["edge_attr_views"] = {
            "retweet_all": edge_attr,
            "temporal_history": hist_edge_attr,
        }
        graph_obj["edge_attr_feature_names_views"] = {
            "retweet_all": EDGE_FEATURE_NAMES,
            "temporal_history": EDGE_FEATURE_NAMES,
        }
        graph_obj["target_edge_index_views"] = {
            "temporal_new": target_new_edge_index,
        }
        graph_obj["future_edge_index"] = target_new_edge_index

        temporal_stats = {
            "history_fraction": args.history_fraction,
            "future_target_mode": args.future_target_mode,
            "history_rows": int(len(hist_rt)),
            "future_rows": int(len(fut_rt)),
            "history_edges": int(hist_edge_index.shape[1]),
            "future_edges": int(len(fut_pairs)),
            "future_overlap_edges": int(len(hist_pairs & fut_pairs)),
            "future_target_edges": int(target_new_edge_index.shape[1]),
        }

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = EDGE_FEATURE_NAMES
    data.user_ids = list(user_ids)
    graph_obj["data"] = data

    torch.save(graph_obj, args.out)

    meta = {
        "csv_glob": args.csv,
        "max_files": args.max_files,
        "nodes": int(len(user_ids)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "label_count": int(len(label_names)),
        "labeled_nodes": int(labeled_nodes),
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        "embedding_dim": emb_stats["embedding_dim"],
        "embedding_matched_users": emb_stats["matched_users"],
        "pseudo_political_labels": bool(args.pseudo_political_labels),
        "pseudo_label_margin": int(args.pseudo_label_margin),
        "keep_isolates": args.keep_isolates,
        "isolated_nodes_before_drop": isolated_before_drop,
        "isolated_nodes_dropped": isolated_dropped,
        "temporal": temporal_stats,
    }
    meta_path = args.out.replace(".pt", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved graph: {args.out}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()

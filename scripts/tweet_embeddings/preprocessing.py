"""Text assembly, normalization, hashing, and skip classification."""

from __future__ import annotations

import hashlib
import re
from typing import Any
import unicodedata

import pandas as pd

from .constants import URL_TOKEN, USER_TOKEN

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
HANDLE_RE = re.compile(r"(?<![\w@])@[\w_]+")
SPACE_RE = re.compile(r"\s+")
INVALID_TEXT_RE = re.compile(r"(?i)\b(deleted|unavailable|withheld)\b")


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value)
    if text in {"", "nan", "None", "<NA>"}:
        return ""
    return text


def has_value(value: Any) -> bool:
    return bool(coerce_text(value).strip())


def preprocess_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(URL_TOKEN, text)
    text = HANDLE_RE.sub(USER_TOKEN, text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def classify_tweet_type(value: Any) -> str:
    return coerce_text(value).strip().lower()


def assemble_text(row: dict[str, Any]) -> tuple[str, str]:
    """Choose one canonical content string for GNN-oriented tweet semantics."""
    tweet_type = classify_tweet_type(row.get("tweet_type"))
    own_text = coerce_text(row.get("text")).strip()
    rt_text = coerce_text(row.get("rt_text")).strip()
    qtd_text = coerce_text(row.get("qtd_text")).strip()

    is_quote = (
        "quote" in tweet_type
        or has_value(row.get("qtd_tweetid"))
        or bool(qtd_text)
    )
    is_retweet = (
        "retweet" in tweet_type
        or has_value(row.get("rt_tweetid"))
        or own_text.startswith("RT ")
    )

    if is_quote and own_text:
        return own_text, "own_text"
    if is_retweet and rt_text:
        return rt_text, "rt_text"
    if own_text:
        return own_text, "own_text"
    if rt_text:
        return rt_text, "rt_text"
    if qtd_text:
        return qtd_text, "qtd_text_fallback"
    return "", "empty"


def skip_reason_for_text(processed_text: str) -> str | None:
    if not processed_text:
        return "empty_after_preprocessing"
    if INVALID_TEXT_RE.search(processed_text):
        return "deleted_unavailable_withheld"
    return None

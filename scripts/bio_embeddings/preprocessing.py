"""Bio normalization, hashing, and role extraction helpers."""

from __future__ import annotations

from typing import Any

from scripts.tweet_embeddings.preprocessing import (
    coerce_text,
    has_value,
    preprocess_text,
    text_hash,
)

from .constants import (
    ROLE_AUTHOR,
    ROLE_QUOTED_AUTHOR,
    ROLE_RETWEETED_AUTHOR,
    ROLE_RETWEETED_QUOTED_AUTHOR,
)


def normalize_bio_text(value: Any) -> str:
    """Normalize a raw profile bio with the bio-text-v001 policy."""
    return preprocess_text(coerce_text(value))


def bio_hash(normalized_bio_text: str) -> str:
    """Return the content-addressed hash for a normalized bio."""
    return text_hash(normalized_bio_text)


def is_retweet_like_row(row: dict[str, Any]) -> bool:
    tweet_type = coerce_text(row.get("tweet_type")).strip().lower()
    text = coerce_text(row.get("text")).strip()
    return (
        "retweet" in tweet_type
        or has_value(row.get("rt_tweetid"))
        or has_value(row.get("rt_text"))
        or text.startswith("RT ")
    )


def extract_role_observations(row: dict[str, Any]) -> list[dict[str, str]]:
    """Extract role/user/raw-bio observations from one source row."""
    observations = [
        {
            "source_role": ROLE_AUTHOR,
            "userid": coerce_text(row.get("userid")),
            "raw_bio_text": coerce_text(row.get("description")),
        }
    ]
    if (
        is_retweet_like_row(row)
        or has_value(row.get("rt_userid"))
        or has_value(row.get("rt_user_description"))
    ):
        observations.append(
            {
                "source_role": ROLE_RETWEETED_AUTHOR,
                "userid": coerce_text(row.get("rt_userid")),
                "raw_bio_text": coerce_text(row.get("rt_user_description")),
            }
        )
    if has_value(row.get("qtd_userid")) or has_value(row.get("qtd_user_description")):
        observations.append(
            {
                "source_role": (
                    ROLE_RETWEETED_QUOTED_AUTHOR
                    if is_retweet_like_row(row)
                    else ROLE_QUOTED_AUTHOR
                ),
                "userid": coerce_text(row.get("qtd_userid")),
                "raw_bio_text": coerce_text(row.get("qtd_user_description")),
            }
        )
    return observations


def is_missing_userid(value: Any) -> bool:
    text = coerce_text(value).strip()
    return not text

"""Constants for the bio embedding pipeline."""

from scripts.tweet_embeddings.constants import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    OUTPUT_DIM,
)

DEFAULT_OUTPUT_ROOT = (
    "/dataMeR2/phil/data/ukr_rus_twitter/bio_embeddings/"
    "gte-multilingual-base/version=v001"
)

PREPROCESSING_VERSION = "bio-text-v001"

ROLE_AUTHOR = "author"
ROLE_RETWEETED_AUTHOR = "retweeted_author"
ROLE_QUOTED_AUTHOR = "quoted_author"
ROLE_RETWEETED_QUOTED_AUTHOR = "retweeted_quoted_author"

ROLE_ORDER = [
    ROLE_AUTHOR,
    ROLE_RETWEETED_AUTHOR,
    ROLE_QUOTED_AUTHOR,
    ROLE_RETWEETED_QUOTED_AUTHOR,
]

BIO_SOURCE_COLUMNS = [
    "tweetid",
    "date",
    "tweet_type",
    "text",
    "userid",
    "description",
    "rt_userid",
    "rt_user_description",
    "rt_tweetid",
    "rt_text",
    "qtd_userid",
    "qtd_user_description",
    "qtd_tweetid",
    "qtd_text",
]


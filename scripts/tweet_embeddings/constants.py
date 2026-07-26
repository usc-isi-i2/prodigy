"""Constants for the tweet embedding pipeline."""

DEFAULT_INPUT_ROOT = "/dataMeR1/phil/data/ukr_rus_twitter/parquet"
DEFAULT_OUTPUT_ROOT = (
    "/dataMeR1/phil/data/ukr_rus_twitter/tweet_embeddings/"
    "gte-multilingual-base/version=v001"
)
DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-base"
DEFAULT_REVISION = "9bbca17d9273fd0d03d5725c7a4b0f6b45142062"
PREPROCESSING_VERSION = "tweet-text-v001"
OUTPUT_DIM = 768
URL_TOKEN = "<URL>"
USER_TOKEN = "<USER>"

TEXT_COLUMNS = [
    "tweetid",
    "text",
    "lang",
    "date",
    "userid",
    "tweet_type",
    "rt_tweetid",
    "rt_text",
    "qtd_tweetid",
    "qtd_text",
    "reply_statusid",
]

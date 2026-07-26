#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_timestamp,
    to_date,
    explode_outer,
    input_file_name,
)

INPUT_PATH = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"
OUT_BASE = "/scratch1/eibl/data/covid19_twitter/parquet"

spark = (
    SparkSession.builder
    .appName("covid-twitter-json-to-parquet")
    .config("spark.sql.files.ignoreCorruptFiles", "true")
    .config("spark.sql.shuffle.partitions", "800")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# JSONL: one tweet JSON object per line
df = (
    spark.read
    .option("mode", "PERMISSIVE")
    .option("multiLine", "false")
    .json(INPUT_PATH)
    .withColumn("_input_file", input_file_name())
)

# Twitter v1 date format, e.g. Tue Jan 28 22:59:59 +0000 2020
df = (
    df.withColumn(
        "created_ts",
        to_timestamp(col("created_at"), "EEE MMM dd HH:mm:ss Z yyyy")
    )
    .withColumn("date", to_date(col("created_ts")))
)

# ---------------------------------------------------------------------
# 1. Raw nested Parquet: preserves structs/arrays such as user, entities,
#    retweeted_status, place, coordinates, etc.
# ---------------------------------------------------------------------

(
    df.repartition("date")
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .partitionBy("date")
    .parquet(f"{OUT_BASE}/raw_nested")
)

# ---------------------------------------------------------------------
# 2. Thin tweet table: easier and faster for most analysis.
# ---------------------------------------------------------------------

tweets = df.select(
    col("id_str").alias("tweet_id"),
    col("id").alias("tweet_id_num"),
    "created_at",
    "created_ts",
    "date",
    "text",
    "lang",
    "source",
    "truncated",
    "is_quote_status",
    "retweet_count",
    "favorite_count",

    col("user.id_str").alias("user_id"),
    col("user.screen_name").alias("screen_name"),
    col("user.name").alias("user_name"),
    col("user.location").alias("user_location"),
    col("user.description").alias("user_description"),
    col("user.followers_count").alias("followers_count"),
    col("user.friends_count").alias("friends_count"),
    col("user.statuses_count").alias("statuses_count"),
    col("user.favourites_count").alias("favourites_count"),
    col("user.verified").alias("verified"),
    col("user.created_at").alias("user_created_at"),

    col("retweeted_status.id_str").alias("retweeted_tweet_id"),
    col("retweeted_status.user.id_str").alias("retweeted_user_id"),
    col("retweeted_status.user.screen_name").alias("retweeted_screen_name"),
    col("retweeted_status.text").alias("retweeted_text"),

    "entities.hashtags",
    "entities.urls",
    "entities.user_mentions",

    "_input_file",
)

(
    tweets.repartition("date")
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .partitionBy("date")
    .parquet(f"{OUT_BASE}/tweets")
)

# ---------------------------------------------------------------------
# 3. Exploded derived tables.
# ---------------------------------------------------------------------

hashtags = (
    df.select(
        col("id_str").alias("tweet_id"),
        "date",
        explode_outer("entities.hashtags").alias("hashtag")
    )
    .select(
        "tweet_id",
        "date",
        col("hashtag.text").alias("hashtag"),
        col("hashtag.indices").alias("indices"),
    )
)

(
    hashtags
    .where(col("hashtag").isNotNull())
    .repartition("date")
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .partitionBy("date")
    .parquet(f"{OUT_BASE}/hashtags")
)

mentions = (
    df.select(
        col("id_str").alias("tweet_id"),
        "date",
        explode_outer("entities.user_mentions").alias("mention")
    )
    .select(
        "tweet_id",
        "date",
        col("mention.id_str").alias("mentioned_user_id"),
        col("mention.screen_name").alias("mentioned_screen_name"),
        col("mention.name").alias("mentioned_name"),
        col("mention.indices").alias("indices"),
    )
)

(
    mentions
    .where(col("mentioned_user_id").isNotNull())
    .repartition("date")
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .partitionBy("date")
    .parquet(f"{OUT_BASE}/mentions")
)

urls = (
    df.select(
        col("id_str").alias("tweet_id"),
        "date",
        explode_outer("entities.urls").alias("url")
    )
    .select(
        "tweet_id",
        "date",
        col("url.url").alias("url"),
        col("url.expanded_url").alias("expanded_url"),
        col("url.display_url").alias("display_url"),
        col("url.indices").alias("indices"),
    )
)

(
    urls
    .where(col("url").isNotNull())
    .repartition("date")
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .partitionBy("date")
    .parquet(f"{OUT_BASE}/urls")
)

print("Done.")
print(f"Wrote Parquet datasets under: {OUT_BASE}")

spark.stop()
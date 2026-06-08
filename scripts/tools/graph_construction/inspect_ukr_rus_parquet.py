from pathlib import Path
import duckdb

BASE = "/dataMeR2/phil/data/ukr_rus_twitter/parquet"
GLOB = f"{BASE}/*/*.parquet"

con = duckdb.connect()
con.execute("PRAGMA threads=32")

def q(title: str, sql: str, limit: int | None = None):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    if limit is not None:
        sql = f"{sql}\nLIMIT {limit}"
    df = con.execute(sql).df()
    with print_full(df):
        print(df.to_string(index=False))

class print_full:
    def __init__(self, df):
        self.df = df
        self.prev_rows = None
        self.prev_cols = None

    def __enter__(self):
        import pandas as pd
        self.prev_rows = pd.get_option("display.max_rows")
        self.prev_cols = pd.get_option("display.max_columns")
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.max_columns", 200)

    def __exit__(self, exc_type, exc, tb):
        import pandas as pd
        pd.set_option("display.max_rows", self.prev_rows)
        pd.set_option("display.max_columns", self.prev_cols)

con.execute(f"""
CREATE OR REPLACE VIEW tweets AS
SELECT
    *,
    filename,
    regexp_extract(filename, '(\\d{{4}}-\\d{{2}})', 1) AS month_part,

    NULLIF(trim(tweetid), '') AS tweetid_n,
    NULLIF(trim(userid), '') AS userid_n,
    NULLIF(trim(reply_userid), '') AS reply_userid_n,
    NULLIF(trim(rt_userid), '') AS rt_userid_n,
    NULLIF(trim(qtd_userid), '') AS qtd_userid_n,
    NULLIF(trim(mentionid), '') AS mentionid_n,
    NULLIF(trim(tweet_type), '') AS tweet_type_n,
    NULLIF(trim(date), '') AS date_n,
    NULLIF(trim(lang), '') AS lang_n,
    NULLIF(trim(verified), '') AS verified_n,
    NULLIF(trim(sent_vader), '') AS sent_vader_n,
    NULLIF(trim(acc_age), '') AS acc_age_n,

    TRY_CAST(NULLIF(trim(friends_count), '') AS BIGINT) AS friends_count_i,
    TRY_CAST(NULLIF(trim(listed_count), '') AS BIGINT) AS listed_count_i,
    TRY_CAST(NULLIF(trim(followers_count), '') AS BIGINT) AS followers_count_i,
    TRY_CAST(NULLIF(trim(favourites_count), '') AS BIGINT) AS favourites_count_i,
    TRY_CAST(NULLIF(trim(statuses_count), '') AS BIGINT) AS statuses_count_i,
    TRY_CAST(NULLIF(trim(rt_qtd_count), '') AS BIGINT) AS rt_qtd_count_i,
    TRY_CAST(NULLIF(trim(rt_rt_count), '') AS BIGINT) AS rt_rt_count_i,
    TRY_CAST(NULLIF(trim(rt_reply_count), '') AS BIGINT) AS rt_reply_count_i,
    TRY_CAST(NULLIF(trim(rt_fav_count), '') AS BIGINT) AS rt_fav_count_i,
    TRY_CAST(NULLIF(trim(qtd_qtd_count), '') AS BIGINT) AS qtd_qtd_count_i,
    TRY_CAST(NULLIF(trim(qtd_rt_count), '') AS BIGINT) AS qtd_rt_count_i,
    TRY_CAST(NULLIF(trim(qtd_reply_count), '') AS BIGINT) AS qtd_reply_count_i,
    TRY_CAST(NULLIF(trim(qtd_fav_count), '') AS BIGINT) AS qtd_fav_count_i,
    TRY_CAST(NULLIF(trim(sent_vader), '') AS DOUBLE) AS sent_vader_f,
    TRY_CAST(NULLIF(trim(acc_age), '') AS DOUBLE) AS acc_age_f
FROM read_parquet('{GLOB}', filename=true)
""")

q("Row count and file count", """
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT filename) AS n_files,
    COUNT(DISTINCT month_part) AS n_months
FROM tweets
""")

q("Rows by month", """
SELECT
    month_part,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT filename) AS n_files
FROM tweets
GROUP BY 1
ORDER BY 1
""")

q("Sample rows", """
SELECT
    month_part, date, tweetid, userid, screen_name, tweet_type,
    reply_userid, rt_userid, qtd_userid, mentionid,
    lang, verified, friends_count, followers_count, sent_vader, acc_age
FROM tweets
USING SAMPLE 10 ROWS
""")

q("tweet_type distribution", """
SELECT
    COALESCE(tweet_type_n, '<NULL_OR_EMPTY>') AS tweet_type,
    COUNT(*) AS n
FROM tweets
GROUP BY 1
ORDER BY n DESC
""")

q("Interaction coverage", """
SELECT
    COUNT(*) AS n_rows,
    COUNT_IF(userid_n IS NOT NULL) AS rows_with_userid,
    COUNT_IF(reply_userid_n IS NOT NULL) AS rows_with_reply_userid,
    COUNT_IF(rt_userid_n IS NOT NULL) AS rows_with_rt_userid,
    COUNT_IF(qtd_userid_n IS NOT NULL) AS rows_with_qtd_userid,
    COUNT_IF(mentionid_n IS NOT NULL) AS rows_with_mentionid
FROM tweets
""")

q("Distinct user cardinalities", """
SELECT
    COUNT(DISTINCT userid_n) AS src_users,
    COUNT(DISTINCT reply_userid_n) AS reply_targets,
    COUNT(DISTINCT rt_userid_n) AS rt_targets,
    COUNT(DISTINCT qtd_userid_n) AS qtd_targets
FROM tweets
""")

q("Top languages", """
SELECT
    COALESCE(lang_n, '<NULL_OR_EMPTY>') AS lang,
    COUNT(*) AS n
FROM tweets
GROUP BY 1
ORDER BY n DESC
LIMIT 50
""")

q("Verified raw values", """
SELECT
    COALESCE(verified_n, '<NULL_OR_EMPTY>') AS verified,
    COUNT(*) AS n
FROM tweets
GROUP BY 1
ORDER BY n DESC
LIMIT 20
""")

q("Numeric parseability", """
SELECT
    COUNT(*) AS n_rows,

    COUNT_IF(friends_count IS NOT NULL AND friends_count_i IS NULL) AS friends_bad,
    COUNT_IF(listed_count IS NOT NULL AND listed_count_i IS NULL) AS listed_bad,
    COUNT_IF(followers_count IS NOT NULL AND followers_count_i IS NULL) AS followers_bad,
    COUNT_IF(favourites_count IS NOT NULL AND favourites_count_i IS NULL) AS favourites_bad,
    COUNT_IF(statuses_count IS NOT NULL AND statuses_count_i IS NULL) AS statuses_bad,

    COUNT_IF(sent_vader IS NOT NULL AND sent_vader_f IS NULL) AS sent_vader_bad,
    COUNT_IF(acc_age IS NOT NULL AND acc_age_f IS NULL) AS acc_age_bad
FROM tweets
""")

q("Date raw samples", """
SELECT DISTINCT date_n
FROM tweets
WHERE date_n IS NOT NULL
LIMIT 20
""")

q("Null / empty rates for graph-critical columns", """
SELECT * FROM (
    VALUES
      ('tweetid',      COUNT_IF(tweetid_n IS NULL),      COUNT(*)),
      ('userid',       COUNT_IF(userid_n IS NULL),       COUNT(*)),
      ('tweet_type',   COUNT_IF(tweet_type_n IS NULL),   COUNT(*)),
      ('reply_userid', COUNT_IF(reply_userid_n IS NULL), COUNT(*)),
      ('rt_userid',    COUNT_IF(rt_userid_n IS NULL),    COUNT(*)),
      ('qtd_userid',   COUNT_IF(qtd_userid_n IS NULL),   COUNT(*)),
      ('mentionid',    COUNT_IF(mentionid_n IS NULL),    COUNT(*)),
      ('lang',         COUNT_IF(lang_n IS NULL),         COUNT(*)),
      ('verified',     COUNT_IF(verified_n IS NULL),     COUNT(*)),
      ('sent_vader',   COUNT_IF(sent_vader_n IS NULL),   COUNT(*)),
      ('acc_age',      COUNT_IF(acc_age_n IS NULL),      COUNT(*))
) AS t(col, null_or_empty, total)
SELECT
    col,
    null_or_empty,
    total,
    ROUND(100.0 * null_or_empty / total, 2) AS pct_null_or_empty
FROM t
ORDER BY pct_null_or_empty DESC
""")

q("Retweet examples", """
SELECT
    date, tweet_type, userid, screen_name, rt_userid, rt_screen,
    rt_tweetid, rt_text, rt_hashtag, rt_rt_count, rt_fav_count
FROM tweets
WHERE rt_userid_n IS NOT NULL
LIMIT 20
""")

q("Reply examples", """
SELECT
    date, tweet_type, userid, screen_name, reply_userid, reply_screen,
    reply_statusid, text
FROM tweets
WHERE reply_userid_n IS NOT NULL
LIMIT 20
""")

q("Quote examples", """
SELECT
    date, tweet_type, userid, screen_name, qtd_userid, qtd_screen,
    qtd_tweetid, qtd_text, qtd_hashtag, qtd_rt_count, qtd_fav_count
FROM tweets
WHERE qtd_userid_n IS NOT NULL
LIMIT 20
""")

q("Mention raw examples", """
SELECT
    userid, screen_name, mentionid, mentionsn, text
FROM tweets
WHERE mentionid_n IS NOT NULL
LIMIT 20
""")

q("Potential duplicate tweet IDs", """
SELECT
    COUNT(*) AS dup_tweet_rows
FROM (
    SELECT tweetid_n
    FROM tweets
    WHERE tweetid_n IS NOT NULL
    GROUP BY 1
    HAVING COUNT(*) > 1
)
""")

q("Top repeated tweet IDs", """
SELECT
    tweetid_n,
    COUNT(*) AS n
FROM tweets
WHERE tweetid_n IS NOT NULL
GROUP BY 1
HAVING COUNT(*) > 1
ORDER BY n DESC
LIMIT 20
""")

q("Mention field shape examples", """
SELECT
    mentionid,
    mentionsn,
    regexp_matches(mentionid, '^[0-9]+$') AS mentionid_is_scalar_numeric,
    regexp_matches(mentionid, '.*[,\\[\\]\\|].*') AS mentionid_looks_list_like
FROM tweets
WHERE mentionid_n IS NOT NULL
LIMIT 50
""")

q("Hashtag field shape examples", """
SELECT
    hashtag,
    rt_hashtag,
    qtd_hashtag
FROM tweets
WHERE hashtag IS NOT NULL OR rt_hashtag IS NOT NULL OR qtd_hashtag IS NOT NULL
LIMIT 50
""")

q("URL field shape examples", """
SELECT
    urls_list,
    rt_urls_list,
    qtd_urls_list
FROM tweets
WHERE urls_list IS NOT NULL OR rt_urls_list IS NOT NULL OR qtd_urls_list IS NOT NULL
LIMIT 50
""")
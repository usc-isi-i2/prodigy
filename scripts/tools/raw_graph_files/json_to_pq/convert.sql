PRAGMA enable_progress_bar;
PRAGMA threads=16;
SET memory_limit='56GB';

COPY (
    SELECT *
    FROM read_ndjson(
        '/scratch1/eibl/data/covid19_twitter/raw/*/*.json',
        sample_size = 10000,
        maximum_depth = -1,
        maximum_object_size = 104857600,
        ignore_errors = true
    )
    WHERE id_str IS NOT NULL
)
TO '/scratch1/eibl/data/covid19_twitter/parquet/'
(
    FORMAT parquet,
    COMPRESSION zstd,
    COMPRESSION_LEVEL 1,
    ROW_GROUP_SIZE 100000,
    ROW_GROUPS_PER_FILE 8,
    PER_THREAD_OUTPUT true
);
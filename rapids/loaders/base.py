"""Loader protocol for the rapids data pipeline.

A *loader* is responsible for reading raw files into a normalised pandas
DataFrame that the shared graph-building functions can consume.  Any new
dataset format just needs to implement ``RecordLoader`` — nothing else in the
pipeline needs to change.

Minimal implementation example
--------------------------------

    from rapids.loaders.base import RecordLoader
    import pandas as pd

    class MyFormatLoader(RecordLoader):
        def load_dataframe(self, path: str) -> pd.DataFrame:
            # Parse your format here and return a DataFrame with at minimum:
            #   userid (int64), rt_userid (int64), date (str)
            # Optional but used when present:
            #   screen_name, rt_screen, followers_count, verified,
            #   statuses_count, rt_fav_count, rt_reply_count, sent_vader,
            #   hashtag, mentionsn, media_urls, description, urls / urls_list
            ...

        # Optionally override load_records() if you need per-record iteration
        # for the embedding pipeline.  The default delegates to load_dataframe().

Column contract
---------------
The canonical column names expected downstream are:

  Edge endpoints (required):
    userid     – int64 source node id
    rt_userid  – int64 destination node id

  Timestamps (required for temporal views):
    date       – raw date string; format is passed to prepare_retweet_rows()

  Node account features (optional; zeros used when absent):
    followers_count, verified, statuses_count, rt_fav_count, rt_reply_count,
    sent_vader, hashtag, mentionsn, media_urls

  Handle metadata (optional; used for label attachment):
    screen_name, rt_screen, description

  Label signals (optional; dataset-specific):
    urls, urls_list, rt_hashtag, ...

If your dataset uses different column names for the edge endpoints, pass
``src_col`` / ``dst_col`` to ``prepare_retweet_rows()``.
"""
from __future__ import annotations

from typing import Any, Iterator, List

import pandas as pd


class RecordLoader:
    """Base class for dataset loaders.

    Subclass this and implement ``load_dataframe``.  All other methods have
    sensible defaults built on top of it.
    """

    def load_dataframe(self, path: str) -> pd.DataFrame:
        """Load *path* and return a normalised DataFrame.

        Must return a DataFrame whose columns follow the contract documented in
        this module's docstring.  Return an empty DataFrame (``pd.DataFrame()``)
        on unrecoverable errors so the caller can skip the file gracefully.
        """
        raise NotImplementedError

    def load_records(self, path: str) -> List[Any]:
        """Return a list of raw records for the embedding pipeline.

        Each record must be compatible with the ``get_uid`` / ``get_text``
        callables passed to ``run_embedding_pipeline``.

        The default implementation returns the DataFrame rows as dicts, which
        works well when uid and text extraction operate on dict-like objects.
        Override if your format is more naturally iterated as raw objects
        (e.g. JSON tweet dicts) before DataFrame conversion.
        """
        df = self.load_dataframe(path)
        if df.empty:
            return []
        return df.to_dict("records")

    def iter_files(self, paths: List[str]) -> Iterator[tuple[str, pd.DataFrame]]:
        """Yield ``(path, dataframe)`` pairs, skipping empty or failed files."""
        for path in paths:
            try:
                df = self.load_dataframe(path)
            except Exception as exc:
                print(f"[WARN] {path}: {exc}")
                continue
            if not df.empty:
                yield path, df

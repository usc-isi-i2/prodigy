"""JSON / NDJSON loader for the covid19_twitter dataset."""
import json

import pandas as pd

from rapids.loaders.base import RecordLoader

try:
    import orjson

    def _loads(b):
        return orjson.loads(b)

    _binary = True
except ImportError:
    _binary = False


def load_json_items(path: str) -> list:
    """Load a JSON or NDJSON file and return a flat list of tweet dicts.

    Accepts:
    - A JSON array at the top level.
    - A JSON object with a ``statuses`` or ``data`` list key.
    - One JSON object per line (NDJSON / JSONL).
    """
    if _binary:
        with open(path, "rb") as f:
            raw = f.read()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read().encode()

    if not raw.strip():
        return []

    def _parse(b):
        return orjson.loads(b) if _binary else json.loads(b)

    try:
        obj = _parse(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("statuses", "data"):
                if isinstance(obj.get(k), list):
                    return obj[k]
            return [obj]
        return []
    except Exception:
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(_parse(line))
            except Exception:
                pass
        return items


class JsonLoader(RecordLoader):
    """``RecordLoader`` for JSON/NDJSON files (e.g. covid19_twitter).

    ``load_records`` returns the raw tweet dicts directly (more efficient than
    round-tripping through a DataFrame).  ``load_dataframe`` is provided for
    callers that need a tabular view; it flattens the tweet dicts via
    ``pd.json_normalize``.
    """

    def load_records(self, path: str) -> list:  # type: ignore[override]
        return load_json_items(path)

    def load_dataframe(self, path: str) -> pd.DataFrame:
        items = load_json_items(path)
        if not items:
            return pd.DataFrame()
        return pd.json_normalize(items)

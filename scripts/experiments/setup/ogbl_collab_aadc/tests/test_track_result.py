from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "track_result.py"
SPEC = importlib.util.spec_from_file_location("aadc_track", MODULE)
assert SPEC is not None and SPEC.loader is not None
track = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(track)


def test_official_rows() -> None:
    text = "Official: hits@10=0.1, hits@50=0.2\nnoise\nOfficial: hits@10=0.3, hits@50=0.4"
    assert track.official_rows(text) == [
        {"hits@10": 0.1, "hits@50": 0.2},
        {"hits@10": 0.3, "hits@50": 0.4},
    ]

#!/usr/bin/env python3
"""Dataset-local wrapper for the parquet-backed covid19_twitter graph builder."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.graph_construction.generate_covid19_twitter_retweet_graph_from_parquet import main


if __name__ == "__main__":
    main()

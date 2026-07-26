#!/usr/bin/env python3
"""Generate deterministic bio embedding shards for the Ukraine-Russia corpus."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bio_embeddings.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

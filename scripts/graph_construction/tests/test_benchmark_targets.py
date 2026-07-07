"""Unit tests for scripts/graph_construction/benchmark_targets.py.

Run directly (no pytest required):
    /opt/homebrew/bin/python3.11 scripts/graph_construction/tests/test_benchmark_targets.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_targets as bt  # noqa: E402


def test_profile_targets_alignment_and_missing():
    user_ids = [10, 20, 30]
    raw = {
        10: {
            "followers_count": 100,
            "friends_count": 50,
            "statuses_count": 4000,
            "favourites_count": 12,
            "listed_count": 3,
            "created_at": "Wed Oct 10 20:19:24 +0000 2018",
        },
        # user 20 missing entirely -> all NaN
        30: {"followers_count": -5, "statuses_count": None, "created_at": "not-a-date"},
    }
    ref = datetime(2020, 10, 10, tzinfo=timezone.utc)
    targets, stats = bt.build_profile_node_targets(user_ids, raw, reference_date=ref)

    assert list(targets.keys())[:5] == list(bt.PROFILE_COUNT_FIELDS)
    assert "account_age_days" in targets
    for name, t in targets.items():
        assert t.shape == (3,), name

    fol = targets["followers_count"].numpy()
    assert fol[0] == 100.0
    assert np.isnan(fol[1])           # missing user
    assert np.isnan(fol[2])           # negative -> NaN

    age = targets["account_age_days"].numpy()
    assert abs(age[0] - 731.0) < 1.5  # ~2 years (2018-10-10 -> 2020-10-10)
    assert np.isnan(age[1])
    assert np.isnan(age[2])           # unparseable creation date

    assert stats["coverage"]["followers_count"] == 1
    print("ok: profile targets alignment + missing handling")


def test_static_split_no_undirected_leakage():
    # Reciprocated pair (0,1)/(1,0) plus a chain, 8 undirected pairs total.
    edges = [
        (0, 1), (1, 0), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 7), (7, 8),
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    split = bt.build_static_edge_split(edge_index, holdout_frac=0.25, seed=0)

    bg = set(map(tuple, split.background_edge_index.t().tolist()))
    ho = set(map(tuple, split.holdout_edge_index.t().tolist()))

    # Partition: every original edge is in exactly one view.
    assert len(bg) + len(ho) == len({*map(tuple, edge_index.t().tolist())})
    assert bg.isdisjoint(ho)

    # No undirected pair straddles: reverse of a held-out edge not in background.
    for u, v in ho:
        assert (v, u) not in bg, f"undirected leak: ({u},{v}) held out but ({v},{u}) in background"

    # Mask is consistent with the returned background edges.
    assert int(split.background_mask.sum()) == split.background_edge_index.shape[1]
    assert split.stats["holdout_edges"] == len(ho)
    print("ok: static split partitions edges with no undirected leakage")


def test_static_split_reproducible():
    edge_index = torch.randint(0, 50, (2, 400))
    a = bt.build_static_edge_split(edge_index, holdout_frac=0.15, seed=7)
    b = bt.build_static_edge_split(edge_index, holdout_frac=0.15, seed=7)
    assert torch.equal(a.holdout_edge_index, b.holdout_edge_index)
    print("ok: static split is reproducible under a fixed seed")


def test_attach_creates_views():
    graph_obj: dict = {}
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    split = bt.build_static_edge_split(edge_index, holdout_frac=0.25, seed=1)
    targets, _ = bt.build_profile_node_targets([0, 1, 2, 3], {})
    edge_attr = torch.arange(4, dtype=torch.float).reshape(-1, 1)

    bt.attach_benchmark_targets(
        graph_obj,
        node_targets=targets,
        static_split=split,
        edge_attr=edge_attr,
        edge_attr_feature_names=["n_retweets"],
    )
    assert graph_obj["node_target_names"] == list(bt.PROFILE_TARGET_NAMES)
    assert "static_background" in graph_obj["edge_index_views"]
    assert "static_holdout" in graph_obj["target_edge_index_views"]
    assert graph_obj["edge_attr_views"]["static_background"].shape[0] == int(split.background_mask.sum())
    print("ok: attach_benchmark_targets creates node_targets + static views")


if __name__ == "__main__":
    test_profile_targets_alignment_and_missing()
    test_static_split_no_undirected_leakage()
    test_static_split_reproducible()
    test_attach_creates_views()
    print("\nAll benchmark_targets tests passed.")

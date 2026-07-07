"""Unit test for node-target regression selection in data.midterm.

Covers _select_target_from_feature reading from graph.node_targets (not x),
the log1p transform, and NaN preservation. Importing data.midterm pulls in
torch_sparse, so run in the ``prodigy`` env:

    python -m data.tests.test_regression_target     # from repo root
"""

from __future__ import annotations

import math

import torch
from torch_geometric.data import Data

from data.midterm import _apply_target_transform, _select_target_from_feature


def _graph_with_targets():
    g = Data(x=torch.randn(4, 8), edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=4)
    g.feature_names = [f"bio_emb_{i}" for i in range(8)]
    g.label_names = []
    g.node_targets = {
        "followers_count": torch.tensor([100.0, float("nan"), 0.0, 999.0]),
        "account_age_days": torch.tensor([365.0, 30.0, float("nan"), 1.0]),
    }
    return g


def test_selects_from_node_targets_without_touching_x():
    g = _graph_with_targets()
    x_before = g.x.clone()
    out = _select_target_from_feature(g, "followers_count", transform="none")
    assert out.label_type == "regression"
    assert out.label_names == ["followers_count"]
    assert torch.equal(out.x, x_before)                 # x untouched
    assert out.x.shape[1] == 8                          # no column removed
    y = out.y
    assert y[0].item() == 100.0 and y[3].item() == 999.0
    assert math.isnan(y[1].item())                      # missing preserved
    print("ok: regression target read from node_targets, x untouched, NaN preserved")


def test_log1p_transform():
    g = _graph_with_targets()
    out = _select_target_from_feature(g, "followers_count", transform="log1p")
    assert abs(out.y[0].item() - math.log1p(100.0)) < 1e-5
    assert out.y[2].item() == 0.0                        # log1p(0) == 0
    assert math.isnan(out.y[1].item())
    print("ok: log1p transform applied to node target")


def test_apply_transform_validation():
    y = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(_apply_target_transform(y, "none"), y)
    assert torch.allclose(_apply_target_transform(y, "log1p"), torch.log1p(y))
    try:
        _apply_target_transform(y, "sqrt")
    except ValueError:
        print("ok: unknown transform rejected")
        return
    raise AssertionError("expected ValueError for unknown transform")


def test_unknown_target_lists_available():
    g = _graph_with_targets()
    try:
        _select_target_from_feature(g, "not_a_field")
    except ValueError as exc:
        assert "followers_count" in str(exc)             # node_targets listed
        print("ok: unknown target error lists available node_targets")
        return
    raise AssertionError("expected ValueError for unknown target")


if __name__ == "__main__":
    test_selects_from_node_targets_without_touching_x()
    test_log1p_transform()
    test_apply_transform_validation()
    test_unknown_target_lists_available()
    print("\nAll regression_target tests passed.")

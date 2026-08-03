from pathlib import Path

import pytest
import torch

from scripts.graph_construction.merge_disjoint_graph_pt import _merge_graphs


def _graph(offset):
    return {
        "x": torch.full((3, 2), float(offset)),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_attr": torch.ones(2, 1),
        "feature_names": ["a", "b"],
        "edge_attr_feature_names": ["weight"],
        "label_names": ["label"],
        "user_ids": [f"u{offset}-{i}" for i in range(3)],
        "edge_index_views": {
            "static_background": torch.tensor([[0], [1]], dtype=torch.long),
            "source_only": torch.tensor([[1], [2]], dtype=torch.long),
        },
        "target_edge_index_views": {
            "static_holdout": torch.tensor([[1], [2]], dtype=torch.long),
        },
    }


def test_drop_edge_features_can_preserve_only_requested_split_views():
    merged = _merge_graphs(
        ["a", "b"],
        [Path("a.pt"), Path("b.pt")],
        [_graph(0), _graph(1)],
        drop_edge_features=True,
        preserve_edge_views=["static_background"],
        preserve_target_edge_views=["static_holdout"],
    )
    assert merged["edge_attr"] is None
    assert set(merged["edge_index_views"]) == {"static_background"}
    assert set(merged["target_edge_index_views"]) == {"static_holdout"}
    assert merged["edge_index_views"]["static_background"].tolist() == [[0, 3], [1, 4]]
    assert merged["target_edge_index_views"]["static_holdout"].tolist() == [[1, 4], [2, 5]]
    assert merged["graph_id"].tolist() == [0, 0, 0, 1, 1, 1]


def test_requested_view_missing_from_one_input_fails():
    second = _graph(1)
    del second["target_edge_index_views"]["static_holdout"]
    with pytest.raises(ValueError, match="missing requested.*static_holdout"):
        _merge_graphs(
            ["a", "b"],
            [Path("a.pt"), Path("b.pt")],
            [_graph(0), second],
            drop_edge_features=True,
            preserve_target_edge_views=["static_holdout"],
        )

"""Tests for the shared Cora/PubMed GTE graph converter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.graph_construction.generate_tag_citation_graph import (
    DATASETS,
    OUTPUT_DIM,
    assemble_graph_artifact,
    normalize_raw_texts,
)
from data import tag_citation


def test_normalize_raw_texts_preserves_order_and_cleans_whitespace():
    assert normalize_raw_texts({"raw_texts": ["  first\n paper ", None, "third"]}) == [
        "first paper",
        "",
        "third",
    ]


def test_assemble_graph_artifact_aligns_fields_and_prevents_reverse_leakage():
    # Seven nodes let the synthetic fixture exercise every canonical Cora label.
    source = SimpleNamespace(
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
            ],
            dtype=torch.long,
        ),
        y=torch.arange(7),
        train_mask=torch.tensor([1, 1, 0, 0, 0, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 1, 1, 0, 0, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.bool),
        keys=lambda: ["edge_index", "y", "train_mask", "val_mask", "test_mask"],
    )
    texts = [f"paper {index}" for index in range(7)]
    embeddings = torch.nn.functional.normalize(torch.randn(7, OUTPUT_DIM), dim=1)

    with patch.dict(DATASETS["cora"], {"nodes": 7}):
        artifact, metadata = assemble_graph_artifact(
            "cora", source, texts, embeddings, static_holdout_frac=0.34, seed=3
        )

    assert artifact["x"].shape == (7, OUTPUT_DIM)
    assert artifact["x"].dtype == torch.float32
    assert artifact["raw_texts"] == texts
    assert artifact["label_names"] == DATASETS["cora"]["label_names"]
    assert torch.equal(artifact["train_mask"], source.train_mask)
    assert metadata["num_nodes"] == 7
    assert metadata["num_edges"] == 12

    background = {
        tuple(sorted(edge))
        for edge in artifact["edge_index_views"]["static_background"].t().tolist()
    }
    holdout = {
        tuple(sorted(edge))
        for edge in artifact["target_edge_index_views"]["static_holdout"].t().tolist()
    }
    assert background
    assert holdout
    assert background.isdisjoint(holdout)

def test_source_hashes_are_pinned_sha256_values():
    for spec in DATASETS.values():
        assert set(spec["files"]) == {"processed_data.pt", "raw_texts.pt"}
        assert all(len(digest) == 64 for digest in spec["files"].values())


def test_static_link_loader_defaults_to_leakage_free_background():
    with patch.object(tag_citation, "_get_dataset", return_value="dataset") as loader:
        result = tag_citation.get_cora_dataset(
            "/tmp/graphs",
            task_name="static_link_prediction",
            edge_view="default",
        )
    assert result == "dataset"
    assert loader.call_args.kwargs["edge_view"] == "static_background"


if __name__ == "__main__":
    test_normalize_raw_texts_preserves_order_and_cleans_whitespace()
    test_assemble_graph_artifact_aligns_fields_and_prevents_reverse_leakage()
    test_source_hashes_are_pinned_sha256_values()
    test_static_link_loader_defaults_to_leakage_free_background()
    print("All TAG citation graph tests passed.")

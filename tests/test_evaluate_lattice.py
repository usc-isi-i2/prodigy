from pathlib import Path

import pytest
import torch

from mixture_scaling.evaluate_lattice import (
    CLS_TARGETS, LP_TARGETS, cached_lp_views, load_pair_module,
)


def test_target_contracts():
    assert len(CLS_TARGETS) == 5
    assert len(LP_TARGETS) == 6
    assert "facebook_page_reference" in CLS_TARGETS
    assert "facebook_page_reference" in LP_TARGETS


def test_actual_repaired_pair_evaluator_imports_when_repo_is_available():
    root = Path(__file__).resolve().parents[2] / "prodigy"
    if root.is_dir():
        module = load_pair_module(root)
        assert module.self_test(verbose=False)


def test_cached_lp_views_reconstructs_symmetric_background(tmp_path):
    cache = tmp_path / "_cache"
    cache.mkdir()
    train = torch.tensor([[0, 2], [1, 3]])
    holdout = torch.tensor([[4], [5]])
    torch.save({"train_edges": train, "validation_edges": holdout}, cache / "tiny_edge_split_s0.pt")
    background, actual_holdout = cached_lp_views(tmp_path, "tiny", 0)
    assert torch.equal(background, torch.cat((train, train.flip(0)), dim=1))
    assert torch.equal(actual_holdout, holdout)


def test_cached_lp_views_requires_canonical_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="canonical LP edge split"):
        cached_lp_views(tmp_path, "tiny", 0)

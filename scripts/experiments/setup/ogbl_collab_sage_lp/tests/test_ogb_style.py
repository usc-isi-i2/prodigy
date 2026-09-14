from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).parents[1] / "run_ogb_style.py"
SPEC = importlib.util.spec_from_file_location("ogbl_collab_sage_ogb_style", MODULE)
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def test_official_parameter_count() -> None:
    encoder, predictor = run.SAGE(128), run.LinkPredictor()
    count = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in predictor.parameters())
    assert count == run.EXPECTED_PARAMETERS == 460_289


def test_predictor_is_order_symmetric() -> None:
    predictor = run.LinkPredictor()
    left, right = torch.randn(4, 256), torch.randn(4, 256)
    assert torch.equal(predictor(left, right), predictor(right, left))

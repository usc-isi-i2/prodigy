from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "make_model_list.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("sat_h2_make_model_list", MODULE_PATH)
model_list = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(model_list)


def raw_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_digest_ignores_torch_archive_metadata(tmp_path: Path) -> None:
    state = {"model": {"weight": torch.arange(12).reshape(3, 4).float()}}
    left = tmp_path / "left.ckpt"
    right = tmp_path / "right.ckpt"
    torch.save(state, left)
    torch.save(state, right)

    assert raw_digest(left) != raw_digest(right)
    assert model_list.model_digest(left) == model_list.model_digest(right)


def test_model_digest_detects_tensor_difference(tmp_path: Path) -> None:
    left = tmp_path / "left.ckpt"
    right = tmp_path / "right.ckpt"
    torch.save({"model": {"weight": torch.zeros(2)}}, left)
    torch.save({"model": {"weight": torch.ones(2)}}, right)

    assert model_list.model_digest(left) != model_list.model_digest(right)

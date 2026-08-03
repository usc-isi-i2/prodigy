from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "make_model_list.py"
SPEC = importlib.util.spec_from_file_location("nmld_h2_make_model_list", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_registered_plan_has_40_logical_rows_and_39_physical_models():
    rows = MODULE.logical_rows()
    physical = MODULE.physical_models(rows)
    assert len(rows) == 40
    assert len(physical) == 39
    assert Counter(row["variant"] for row in rows) == {
        "matched40k": 8,
        "sequential": 8,
        "split": 8,
        "fixed10k": 16,
    }
    assert Counter(row["variant"] for row in physical) == {
        "matched40k": 8,
        "sequential": 8,
        "split": 8,
        "fixed10k": 15,
    }


def test_fixed_exposure_steps_and_shared_all8_are_explicit():
    rows = [row for row in MODULE.logical_rows() if row["variant"] == "fixed10k"]
    for row in rows:
        assert row["checkpoint_step"] == row["rung"] * 10_000
    all8 = [row for row in rows if row["rung"] == 8]
    assert len(all8) == 2
    assert all8[0]["model_key"] == all8[1]["model_key"]


def test_every_canonical_trajectory_is_complete():
    grouped = Counter((row["variant"], row["order"]) for row in MODULE.logical_rows())
    assert grouped == {
        ("matched40k", "A"): 8,
        ("sequential", "A"): 8,
        ("split", "A"): 8,
        ("fixed10k", "A"): 8,
        ("fixed10k", "C"): 8,
    }

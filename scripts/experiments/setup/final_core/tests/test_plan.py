from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_plan import ORDERS, build_models, select_validation_checkpoint  # noqa: E402


def test_final_core_has_31_unique_models_and_36_aliases():
    models = build_models()
    assert len(models) == 31
    assert len({frozenset(model.sources) for model in models}) == 31
    assert sum(len(model.aliases) for model in models) == 36


def test_frozen_orders_and_reuse():
    assert ORDERS["A"][0] == "ukr_rus"
    assert ORDERS["B"][0] == "ukr_rus_suspended"
    assert ORDERS["C"] == tuple(reversed(ORDERS["B"]))
    models = build_models()
    assert sum(model.model_id == "all9" for model in models) == 1
    assert all(any(alias.startswith("specialist:") for alias in model.aliases)
               for model in models if len(model.sources) == 1)


def test_validation_selection_uses_score_then_earliest_step():
    rows = [
        {"checkpoint_step": 100, "score": 0.70},
        {"checkpoint_step": 300, "score": 0.75},
        {"checkpoint_step": 900, "score": 0.75},
        {"checkpoint_step": 2500, "score": 0.72},
    ]
    assert select_validation_checkpoint(rows)["checkpoint_step"] == 300


def test_validation_selection_requires_the_frozen_schedule():
    rows = [{"checkpoint_step": step, "score": 0.5} for step in (100, 300, 2500)]
    try:
        select_validation_checkpoint(rows)
    except ValueError as error:
        assert "expected one result" in str(error)
    else:
        raise AssertionError("missing checkpoint must fail")

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path


ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "assemble_results.py"
SETUP_PATH = (
    Path(__file__).resolve().parents[3]
    / "setup/nm_ladder_downstream_nhop2/make_model_list.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_module("nmld_h2_assemble", ANALYSIS_PATH)
PLAN = load_module("nmld_h2_plan_for_assemble", SETUP_PATH)


def synthetic_cells():
    rows = PLAN.logical_rows()
    models = {row["model_key"] for row in rows}
    classification = {
        (model, dataset): {"roc_auc": 0.6, "accuracy": 0.55, "f1": 0.5}
        for model in models
        for dataset in ANALYSIS.CLASSIFICATION_DATASETS
    }
    static_lp = {
        (model, dataset): {"auc": 0.7, "average_precision": 0.65, "hits_at_50": 0.2}
        for model in models
        for dataset in ANALYSIS.STATIC_LP_DATASETS
    }
    return rows, classification, static_lp


def test_full_matrix_counts_and_entry_events():
    rows, classification, static_lp = synthetic_cells()
    long_rows, missing = ANALYSIS.build_long_rows(rows, classification, static_lp)
    assert missing == []
    assert len(long_rows) == 40 * (4 + 5) * 3

    jumps = ANALYSIS.entry_jumps(long_rows)
    assert len(jumps) == 40
    assert Counter(row["task"] for row in jumps) == {
        "classification": 19,
        "static_lp": 21,
    }


def test_control_pairs_cover_three_order_a_variants():
    rows, classification, static_lp = synthetic_cells()
    long_rows, _ = ANALYSIS.build_long_rows(rows, classification, static_lp)
    paired = ANALYSIS.pair_to_control(long_rows)
    assert len(paired) == 3 * 8 * (4 + 5)
    assert Counter(row["variant"] for row in paired) == {
        "sequential": 72,
        "split": 72,
        "fixed10k": 72,
    }


def test_wide_tables_expand_shared_fixed_all8_logically():
    rows, classification, static_lp = synthetic_cells()
    long_rows, _ = ANALYSIS.build_long_rows(rows, classification, static_lp)
    classification_wide = ANALYSIS.wide_rows(
        long_rows, "classification", ANALYSIS.CLASSIFICATION_DATASETS
    )
    static_wide = ANALYSIS.wide_rows(
        long_rows, "static_lp", ANALYSIS.STATIC_LP_DATASETS
    )
    assert len(classification_wide) == 40
    assert len(static_wide) == 40

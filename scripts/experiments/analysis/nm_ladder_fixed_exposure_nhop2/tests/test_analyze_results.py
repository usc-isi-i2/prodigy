from pathlib import Path

import pandas as pd

from scripts.experiments.analysis.nm_ladder_fixed_exposure_nhop2.analyze_results import (
    assemble_logical,
    compare_matched40k_h1_order_a,
    compare_matched40k_h2_order_a,
    load_plan,
    make_adjacent_deltas,
    make_rung_summary,
    two_sided_sign_p,
    validate_raw,
)


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[3]
RAW = HERE / "data" / "raw_metrics.csv"
MANIFEST = (
    REPO / "scripts" / "experiments" / "setup"
    / "nm_ladder_fixed_exposure_nhop2" / "manifest.tsv"
)


def test_complete_matrix_and_shared_all8_expansion():
    raw = pd.read_csv(RAW)
    validate_raw(raw)
    logical = assemble_logical(raw, load_plan(MANIFEST))

    assert len(raw) == 120
    assert len(logical) == 128
    assert logical.groupby(["order", "rung"]).dataset.nunique().eq(8).all()

    a8 = logical[(logical.order == "A") & (logical.rung == 8)].sort_values("dataset")
    c8 = logical[(logical.order == "C") & (logical.rung == 8)].sort_values("dataset")
    assert (a8.model.to_numpy() == c8.model.to_numpy()).all()
    assert (a8.test_roc_auc.to_numpy() == c8.test_roc_auc.to_numpy()).all()
    assert c8.shared_all8_artifact.all()


def test_event_roles_and_entry_staircase():
    logical = assemble_logical(pd.read_csv(RAW), load_plan(MANIFEST))
    deltas = make_adjacent_deltas(logical)
    entry = deltas[deltas.role == "newcomer"]

    assert len(deltas) == 112
    assert len(entry) == 14
    assert (entry.delta_auc > 0).all()
    assert two_sided_sign_p(14, 14) == 2 / 2**14


def test_rung_summary_has_two_complete_trajectories():
    logical = assemble_logical(pd.read_csv(RAW), load_plan(MANIFEST))
    summary = make_rung_summary(logical)
    assert len(summary) == 16
    assert summary.groupby("order").rung.nunique().to_dict() == {"A": 8, "C": 8}
    assert summary[summary.rung == 8].held_out_mean_auc.isna().all()


def test_matched40k_h2_comparison_is_complete_and_protocol_matched():
    logical = assemble_logical(pd.read_csv(RAW), load_plan(MANIFEST))
    matched40k_path = (
        REPO / "scripts" / "experiments" / "analysis" / "nm_ladder_nhop2"
        / "data" / "nm_ladder_nhop2_order_A.csv"
    )
    comparison = compare_matched40k_h2_order_a(logical, matched40k_path)
    assert len(comparison) == 64
    assert comparison.is_entry_cell.sum() == 7
    assert comparison.comparison_scope.str.contains("same fair-two-hop sampler").all()
    assert set(comparison.matched40k_total_steps) == {40_000}
    rung1 = comparison[comparison.rung == 1]
    assert abs(rung1.auc_difference.mean() - 0.0016483) < 1e-6


def test_matched40k_h1_comparison_is_preserved_and_cross_protocol():
    logical = assemble_logical(pd.read_csv(RAW), load_plan(MANIFEST))
    matched40k_path = (
        REPO / "scripts" / "experiments" / "analysis" / "nm_ladder"
        / "data" / "nm_ladder_full.csv"
    )
    comparison = compare_matched40k_h1_order_a(logical, matched40k_path)
    assert len(comparison) == 64
    assert comparison.is_entry_cell.sum() == 7
    assert comparison.comparison_scope.str.contains("cross-protocol").all()
    assert "matched40k_h1_auc" in comparison
    assert abs(comparison.auc_difference.mean() - 0.0032) < 5e-5

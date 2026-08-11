from scripts.experiments.setup.identity_overlap_audit.audit_identity_overlap import (
    overlap_metrics,
    pair_status,
)


def test_overlap_metrics() -> None:
    metrics = overlap_metrics(10, 20, 5)
    assert metrics == {
        "fraction_a": 0.5,
        "fraction_b": 0.25,
        "fraction_smaller": 0.5,
        "jaccard": 0.2,
    }


def test_pair_status_keeps_missingness_explicit() -> None:
    assert pair_status(
        "full_graph_global_twitter_id", "full_graph_global_twitter_id"
    )[0] == "exact"
    assert pair_status(
        "full_graph_global_twitter_id", "partial_global_twitter_id_array"
    )[0] == "partial_exact"
    assert pair_status(
        "full_graph_global_twitter_id", "not_measurable_row_indices_only"
    )[0] == "not_measurable"
    assert pair_status(
        "full_graph_global_twitter_id", "incompatible_platform"
    )[0] == "incompatible_platform"

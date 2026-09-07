import pandas as pd

from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_schedule_health import (
    correlations,
)


def test_correlations_tracks_paired_health_and_accuracy():
    frame = pd.DataFrame({
        "delta_u1_agreement": [-.3, -.1, .1, .3],
        "delta_accuracy": [-.2, -.05, .02, .2],
        "delta_roc_auc": [-.4, -.2, .2, .4],
    })
    rows = correlations(frame, "test")
    assert {row["metric"] for row in rows} == {"accuracy", "roc_auc"}
    assert all(row["spearman"] == 1 for row in rows)
    assert all(row["same_sign_fraction"] == 1 for row in rows)

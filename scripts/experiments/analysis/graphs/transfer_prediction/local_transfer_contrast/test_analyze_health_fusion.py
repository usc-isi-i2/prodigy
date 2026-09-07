import pandas as pd

from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_health_fusion import (
    crossed_bootstrap,
    gain_summary,
    parse_seed_input,
)


def test_parse_seed_input():
    seed, path = parse_seed_input("2=/tmp/export")
    assert seed == 2
    assert str(path) == "/tmp/export"


def test_constant_crossed_fusion_gain():
    rows = []
    for stream in ("original", "fresh"):
        for target in ("a", "b"):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "stream": stream,
                        "target": target,
                        "seed": seed,
                        "method": "trace_health_fusion",
                        "reference": "equal_probability",
                        "target_supported_at_0_55": target == "a",
                        "delta_accuracy": .02,
                        "delta_roc_auc": .01,
                    }
                )
    frame = pd.DataFrame(rows)
    value, interval = crossed_bootstrap(
        frame[frame.stream.eq("fresh")], "delta_accuracy", seed=1, draws=100
    )
    assert abs(value - .02) < 1e-12
    assert abs(interval[0] - .02) < 1e-12
    summary = gain_summary(frame, draws=100)
    row = summary[
        summary.stream.eq("fresh")
        & summary.scope.eq("supported_targets")
        & summary.metric.eq("roc_auc")
    ].iloc[0]
    assert abs(row.mean_delta - .01) < 1e-12
    assert row.cell_wins == 3

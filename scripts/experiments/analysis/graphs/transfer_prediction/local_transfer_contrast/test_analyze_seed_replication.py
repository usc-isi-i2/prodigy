import pandas as pd

from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_seed_replication import (
    centered_correlation,
    parse_seed_path,
)


def test_parse_seed_path():
    seed, path = parse_seed_path("2=/tmp/result")
    assert seed == 2
    assert str(path) == "/tmp/result"


def test_centered_correlation_removes_target_level_shift():
    frame = pd.DataFrame({
        "stream": ["fresh"] * 12,
        "target": ["a"] * 6 + ["b"] * 6,
        "seed": [0, 0, 0, 1, 1, 1] * 2,
        "u1_agreement_rate": [.1, .2, .3, .3, .4, .5, .5, .6, .7, .7, .8, .9],
        "accuracy": [.2, .4, .6, .3, .5, .7, .1, .3, .5, .2, .4, .6],
    })
    result = centered_correlation(frame, "fresh", "accuracy")
    assert result["scope"] == "target_seed_centered_pool"
    assert result["spearman"] > .9

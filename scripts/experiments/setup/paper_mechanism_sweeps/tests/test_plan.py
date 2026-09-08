from pathlib import Path

import yaml

from scripts.experiments.setup.paper_mechanism_sweeps.plan import (
    ARMS,
    CHECKPOINT_STEPS,
    SEEDS,
    write_configs,
)


def test_mechanism_plan_is_exact_and_matched(tmp_path: Path):
    paths = write_configs(tmp_path)
    assert len(paths) == 6
    rows = [yaml.safe_load(path.read_text()) for path in paths]
    assert {row["neighbor_sampling_cross_source_prob"] for row in rows if row["emb_dim"] == 256} == {
        0.0, 0.1, 0.25, 0.5, 1.0
    }
    assert {row["emb_dim"] for row in rows} == {256, 512}
    assert all(row["dataset_len_cap"] * row["epochs"] == 10_000 for row in rows)
    assert all(row["campaign_eval_interval"] == 2_000 for row in rows)
    assert all(row["early_stopping_patience"] > len(CHECKPOINT_STEPS) for row in rows)
    assert all("twibot20" not in row["neighbor_sampling_source_subset"] for row in rows)
    assert len(ARMS) * len(SEEDS) * len(CHECKPOINT_STEPS) == 90

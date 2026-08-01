from pathlib import Path
import sys


SETUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SETUP))

import make_configs  # noqa: E402
import make_model_list  # noqa: E402


def test_plan_has_24_rows_and_21_unique_models():
    rows = make_configs.plan()
    unique = make_configs.unique_rows(rows)
    assert len(rows) == 24
    assert len(unique) == 21
    assert len({row["prefix"] for row in unique}) == 21
    assert len({frozenset(row["sources"]) for row in unique}) == 21


def test_phase_counts_and_h2_reuse_only():
    assert len(make_configs.phase_rows("A")) == 8
    assert len(make_configs.phase_rows("robustness")) == 13
    assert len(make_configs.phase_rows("all")) == 21

    rows = make_configs.plan()
    b2 = next(row for row in rows if row["order"] == "B" and row["rung"] == 2)
    b8 = next(row for row in rows if row["order"] == "B" and row["rung"] == 8)
    c8 = next(row for row in rows if row["order"] == "C" and row["rung"] == 8)
    assert b2["prefix"] == "nm_ladder_h2_ordA_r2"
    assert b8["prefix"] == "nm_ladder_h2_ordA_r8"
    assert c8["prefix"] == "nm_ladder_h2_ordA_r8"


def test_every_generated_training_config_locks_protocol():
    files = make_configs.expected_files()
    configs = {
        path: text
        for path, text in files.items()
        if path.name.startswith("train_")
    }
    assert len(configs) == 21
    for text in configs.values():
        assert "n_hop: 2\n" in text
        assert "layers: S,U,M\n" in text
        assert "epochs: 4\n" in text
        assert "checkpoint_step: 10000\n" in text
        assert "neighbor_sampling_episode_source_weighting: balanced\n" in text
        assert "prefix: nm_ladder_h2_" in text


def test_checkpoint_resolver_skips_newer_incomplete_retry(tmp_path):
    prefix = "nm_ladder_h2_ordA_r1"
    complete = tmp_path / f"{prefix}_01_08_2026_10_00_00" / "checkpoint"
    complete.mkdir(parents=True)
    checkpoint = complete / "state_dict_40000.ckpt"
    checkpoint.touch()

    incomplete = tmp_path / f"{prefix}_01_08_2026_11_00_00"
    incomplete.mkdir()
    incomplete.touch()

    assert make_model_list.complete_checkpoint(tmp_path, prefix, 40000) == checkpoint

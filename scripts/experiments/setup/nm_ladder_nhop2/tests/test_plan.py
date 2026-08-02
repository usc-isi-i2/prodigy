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


def test_phase_counts_and_h2m_reuse_only():
    assert len(make_configs.phase_rows("A")) == 8
    assert len(make_configs.phase_rows("robustness")) == 13
    assert len(make_configs.phase_rows("all")) == 21

    rows = make_configs.plan()
    b2 = next(row for row in rows if row["order"] == "B" and row["rung"] == 2)
    b8 = next(row for row in rows if row["order"] == "B" and row["rung"] == 8)
    c8 = next(row for row in rows if row["order"] == "C" and row["rung"] == 8)
    assert b2["prefix"] == "nm_ladder_h2m_ordA_r2"
    assert b8["prefix"] == "nm_ladder_h2m_ordA_r8"
    assert c8["prefix"] == "nm_ladder_h2m_ordA_r8"


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
        assert 'neighbor_sampling_hop_sizes: "9,9"\n' in text
        assert "neighbor_sampling_node_limit: 101\n" in text
        assert "neighbor_matching_walk_hops: 1\n" in text
        assert "layers: S,U,M\n" in text
        assert "gnn_type: sage\n" in text
        assert "n_layer: 1\n" in text
        assert "dropout: 0\n" in text
        assert "epochs: 4\n" in text
        assert "checkpoint_step: 10000\n" in text
        assert "neighbor_sampling_episode_source_weighting: balanced\n" in text
        assert "workers: 2\n" in text
        assert "prefix: nm_ladder_h2m_" in text


def test_smoke_and_eval_use_the_same_compute_matched_sampler():
    files = make_configs.expected_files()
    smoke = files[make_configs.CONFIG_DIR / "smoke_election.yaml"]
    assert "dataset: election2020\n" in smoke
    assert "dataset_len_cap: 20\n" in smoke
    assert 'checkpoint_steps: "0,20"\n' in smoke
    assert 'neighbor_sampling_hop_sizes: "9,9"\n' in smoke
    assert "neighbor_sampling_node_limit: 101\n" in smoke
    assert "neighbor_matching_walk_hops: 1\n" in smoke
    assert "prefix: nm_ladder_h2m_smoke_election\n" in smoke

    eval_script = (SETUP / "eval_ladder_tucker.sh").read_text(encoding="utf-8")
    assert "--n_hop 2" in eval_script
    assert "--neighbor_sampling_hop_sizes 9,9" in eval_script
    assert "--neighbor_sampling_node_limit 101" in eval_script
    assert "--neighbor_matching_walk_hops 1" in eval_script


def test_checkpoint_resolver_skips_newer_incomplete_retry(tmp_path):
    prefix = "nm_ladder_h2m_ordA_r1"
    complete = tmp_path / f"{prefix}_01_08_2026_10_00_00" / "checkpoint"
    complete.mkdir(parents=True)
    checkpoint = complete / "state_dict_40000.ckpt"
    checkpoint.touch()

    incomplete = tmp_path / f"{prefix}_01_08_2026_11_00_00"
    incomplete.mkdir()
    incomplete.touch()

    assert make_model_list.complete_checkpoint(tmp_path, prefix, 40000) == checkpoint

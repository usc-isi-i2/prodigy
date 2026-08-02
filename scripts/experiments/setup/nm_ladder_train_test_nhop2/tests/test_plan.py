from pathlib import Path
import sys

SETUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SETUP))

import make_configs  # noqa: E402
import make_model_list  # noqa: E402


def test_canonical_plan_has_eight_nested_rungs():
    rows = make_configs.plan()
    assert len(rows) == 8
    assert [row["rung"] for row in rows] == list(range(1, 9))
    for previous, current in zip(rows, rows[1:]):
        assert current["sources"][:-1] == previous["sources"]
    assert len({row["prefix"] for row in rows}) == 8


def test_every_config_locks_split_and_fair_two_hop_protocol():
    files = make_configs.expected_files()
    configs = [
        text for path, text in files.items() if path.name.startswith("train_r")
    ]
    assert len(configs) == 8
    for text in configs:
        assert "graph_filename: ukr_rus_covid_midterm_all8_static_split_retweet_graph.pt\n" in text
        assert "edge_view: static_background\n" in text
        assert "target_edge_view: static_holdout\n" in text
        assert "neighbor_matching_edge_split: true\n" in text
        assert "n_hop: 2\n" in text
        assert 'neighbor_sampling_hop_sizes: "9,9"\n' in text
        assert "neighbor_sampling_node_limit: 101\n" in text
        assert "neighbor_matching_walk_hops: 1\n" in text
        assert "epochs: 4\n" in text
        assert "checkpoint_step: 10000\n" in text
        assert "neighbor_sampling_episode_source_weighting: balanced\n" in text


def test_merge_config_preserves_only_named_static_views():
    text = (SETUP / "merge_all8_static_split.yaml").read_text()
    assert "drop_edge_features: true\n" in text
    assert "build_static_split_if_missing: true\n" in text
    assert "static_split_seed: 0\n" in text
    assert "static_holdout_frac: 0.15\n" in text
    assert "preserve_edge_views:\n  - static_background\n" in text
    assert "preserve_target_edge_views:\n  - static_holdout\n" in text
    assert "all8_static_split_retweet_graph.pt\n" in text


def test_eval_forces_same_split_and_sampler_tuple():
    text = (SETUP / "eval_ladder_tucker.sh").read_text()
    for fragment in (
        "--n_hop 2",
        "--neighbor_sampling_hop_sizes 9,9",
        "--neighbor_sampling_node_limit 101",
        "--neighbor_matching_walk_hops 1",
        "--edge_view static_background",
        "--target_edge_view static_holdout",
        "--neighbor_matching_edge_split True",
    ):
        assert fragment in text


def test_checkpoint_resolver_skips_newer_incomplete_retry(tmp_path):
    prefix = "nm_ladder_tts_h2m_r1"
    complete = tmp_path / f"{prefix}_old" / "checkpoint"
    complete.mkdir(parents=True)
    checkpoint = complete / "state_dict_40000.ckpt"
    checkpoint.touch()
    incomplete = tmp_path / f"{prefix}_new"
    incomplete.mkdir()
    assert make_model_list.complete_checkpoint(tmp_path, prefix, 40000) == checkpoint

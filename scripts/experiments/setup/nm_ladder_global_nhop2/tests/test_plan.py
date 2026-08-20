from pathlib import Path
import sys


SETUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SETUP))

import make_configs  # noqa: E402


def test_order_a_seed0_has_eight_nontrivial_finalcore_jobs():
    rows = make_configs.rows()
    assert [row["rung"] for row in rows] == list(range(2, 10))
    assert len({row["model_id"] for row in rows}) == 8
    assert len({row["job_index"] for row in rows}) == 8
    assert rows[-1]["model_id"] == "all9"


def test_configs_match_finalcore_and_only_change_episode_construction():
    files = make_configs.expected_files()
    configs = [text for path, text in files.items() if path.name.startswith("train_")]
    assert len(configs) == 8
    for text in configs:
        assert "graph_filename: ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt" in text
        assert "edge_view: static_train\n" in text
        assert "target_edge_view: static_test\n" in text
        assert "neighbor_matching_edge_split: true\n" in text
        assert "n_hop: 2\n" in text
        assert 'neighbor_sampling_hop_sizes: "9,9"\n' in text
        assert "neighbor_sampling_node_limit: 101\n" in text
        assert "batch_size: 4\n" in text
        assert "learning_rate: 0.002\n" in text
        assert "weight_decay: 0.001\n" in text
        assert "dataset_len_cap: 2500\n" in text
        assert 'checkpoint_steps: "100,300,900,2500"\n' in text
        assert "neighbor_sampling_episode_source: graph_id\n" in text
        assert "neighbor_sampling_episode_source_weighting: proportional\n" in text
        assert "neighbor_sampling_cross_source_prob: 1.0\n" in text


def test_pipeline_uses_two_train_then_evaluate_workers_and_exact_fixed_test():
    pipeline = (SETUP / "run_all_train_tucker.sh").read_text(encoding="utf-8")
    evaluator = (SETUP / "eval_checkpoint_tucker.sh").read_text(encoding="utf-8")
    assert '[[ "${GPU_IDS[*]}" == "0 1" ]]' in pipeline
    assert "worker 0 0" in pipeline
    assert "worker 1 1" in pipeline
    assert "EVAL_START" in pipeline and "TRAIN_DONE" in pipeline
    assert "evaluate_fixed_grid.py" in evaluator
    assert "--batch-size 32 --episode-count 512" in evaluator
    assert "--reference-fingerprints" in evaluator

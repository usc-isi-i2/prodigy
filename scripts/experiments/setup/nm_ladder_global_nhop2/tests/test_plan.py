from pathlib import Path
import sys


SETUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SETUP))

import make_configs  # noqa: E402


def test_seven_nontrivial_order_a_rungs_are_generated():
    assert [row[0] for row in make_configs.RUNGS] == list(range(2, 9))
    assert len({row[3] for row in make_configs.RUNGS}) == 7


def test_global_configs_lock_protocol_and_omit_source_controls():
    files = make_configs.expected_files()
    configs = {
        path: text
        for path, text in files.items()
        if path.name.startswith("train_")
    }
    assert len(configs) == 7
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
        assert "workers: 2\n" in text
        assert "neighbor_sampling_episode_source:" not in text
        assert "neighbor_sampling_episode_source_weighting:" not in text
        assert "neighbor_sampling_source_subset:" not in text


def test_streaming_scripts_hard_restrict_gpu_roles():
    train = (SETUP / "run_all_train_tucker.sh").read_text(encoding="utf-8")
    watch = (SETUP / "watch_and_eval_tucker.sh").read_text(encoding="utf-8")
    evaluate = (SETUP / "eval_checkpoint_tucker.sh").read_text(encoding="utf-8")
    assert '[[ "${GPU}" == "0" ]]' in train
    assert '[[ "${GPU}" == "1" ]]' in watch
    assert '[[ "${GPU}" == "1" ]]' in evaluate
    for script in (watch, evaluate):
        assert "--n_hop 2" in script or "eval_checkpoint_tucker.sh" in script
    assert "10000 20000 30000 40000" in watch
    assert "ALL_DATASETS=" in watch

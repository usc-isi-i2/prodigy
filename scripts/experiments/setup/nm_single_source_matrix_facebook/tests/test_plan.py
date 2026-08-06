from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def test_facebook_config_matches_historical_matrix_protocol():
    text = (HERE / "facebook_page_reference.yaml").read_text()
    assert "graph_filename: page_reference_structural.pt\n" in text
    assert "n_way: 30\n" in text
    assert "n_shots: 3\n" in text
    assert "n_hop: 1\n" in text
    assert "epochs: 5\n" in text
    assert "checkpoint_step: 10000\n" in text
    assert "seed: 0\n" in text


def test_parallel_runner_pins_matrix_shape_and_gpu_split():
    text = (HERE / "run_matrix_tucker.sh").read_text()
    assert 'REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"' in text
    assert "TRAIN_GPU=\"${TRAIN_GPU:-0}\"" in text
    assert "COLUMN_GPUS=\"${COLUMN_GPUS:-1,2,3}\"" in text
    assert "ROW_GPUS=\"${ROW_GPUS:-0,1,2,3}\"" in text
    assert "--datasets facebook_page_reference" in text
    assert "--graph-filenames \"${FB_GRAPH_OVERRIDE}\"" in text
    assert "--nm-n-way 30" in text
    historical = [line for line in text.splitlines() if line.strip().startswith("nm_ss_")]
    assert len(historical) == 8
    assert "nm_ss_facebook_page_reference" in text

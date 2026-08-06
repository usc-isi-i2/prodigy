from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def test_rung9_config_matches_historical_ladder_protocol():
    text = (HERE / "train_ordA_r9.yaml").read_text()
    assert "graph_filename: ukr_rus_covid_midterm_all9_facebook_graph.pt\n" in text
    assert "n_way: 30\n" in text
    assert "n_shots: 3\n" in text
    assert "n_query: 4\n" in text
    assert "n_hop: 1\n" in text
    assert "neighbor_sampling_episode_source: graph_id\n" in text
    assert "neighbor_sampling_episode_source_weighting: balanced\n" in text
    assert "epochs: 5\n" in text
    assert "checkpoint_step: 10000\n" in text
    assert "seed: 0\n" in text


def test_runner_pins_parallel_17_cell_plan():
    text = (HERE / "run_ladder_tucker.sh").read_text()
    assert 'TRAIN_GPU="${TRAIN_GPU:-0}"' in text
    assert 'COLUMN_GPUS="${COLUMN_GPUS:-1,2,3}"' in text
    assert 'ROW_GPUS="${ROW_GPUS:-0,1,2,3}"' in text
    assert "--datasets facebook_page_reference" in text
    assert '--datasets "${DATASETS}"' in text
    assert '--graph-filenames "${FB_GRAPH_OVERRIDE}"' in text
    assert "--nm-n-way 30" in text
    assert text.count('"nm_ladder_ordA_r') >= 8
    assert "nm_ladder_ordA_r9_facebook" in text


def test_order_d_inserts_facebook_at_rung_6_and_reuses_rung_9():
    expected_subsets = {
        6: "ukr_rus,covid,midterm,covid_political,election2020,facebook_page_reference",
        7: "ukr_rus,covid,midterm,covid_political,election2020,facebook_page_reference,ukr_rus_suspended",
        8: "ukr_rus,covid,midterm,covid_political,election2020,facebook_page_reference,ukr_rus_suspended,twibot20",
    }
    for rung, subset in expected_subsets.items():
        text = (HERE / f"train_ordD_r{rung}.yaml").read_text()
        assert "graph_filename: ukr_rus_covid_midterm_all9_facebook_graph.pt\n" in text
        assert f"neighbor_sampling_source_subset: {subset}\n" in text
        assert "n_hop: 1\n" in text
        assert "epochs: 5\n" in text
        assert "checkpoint_step: 10000\n" in text
        assert "seed: 0\n" in text

    runner = (HERE / "run_orderD_tucker.sh").read_text()
    assert 'TRAIN_GPUS="${TRAIN_GPUS:-0 2 3}"' in runner
    assert 'EVAL_GPUS="${EVAL_GPUS:-0,2,3}"' in runner
    assert "CONFIGS=(train_ordD_r6.yaml train_ordD_r7.yaml train_ordD_r8.yaml)" in runner
    assert "3 models x 9 graphs" in runner
    assert "train_ordD_r9.yaml" not in runner

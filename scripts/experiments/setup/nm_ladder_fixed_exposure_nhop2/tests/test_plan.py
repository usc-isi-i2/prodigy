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


def test_phase_counts_and_reuse_are_source_set_safe():
    assert len(make_configs.phase_rows("A")) == 8
    assert len(make_configs.phase_rows("robustness")) == 13
    assert len(make_configs.phase_rows("all")) == 21

    rows = make_configs.plan()
    b2 = next(row for row in rows if row["order"] == "B" and row["rung"] == 2)
    b8 = next(row for row in rows if row["order"] == "B" and row["rung"] == 8)
    c8 = next(row for row in rows if row["order"] == "C" and row["rung"] == 8)
    assert b2["prefix"] == "nm_ladder_fx10k_h2m_ordA_r2"
    assert b2["target_step"] == 20_000
    assert b8["prefix"] == "nm_ladder_fx10k_h2m_ordA_r8"
    assert c8["prefix"] == "nm_ladder_fx10k_h2m_ordA_r8"
    assert b8["target_step"] == c8["target_step"] == 80_000


def test_every_order_uses_the_10k_to_80k_rung_schedule():
    rows = make_configs.plan()
    for order in ("A", "B", "C"):
        order_rows = [row for row in rows if row["order"] == order]
        assert [row["target_step"] for row in order_rows] == [
            10_000, 20_000, 30_000, 40_000,
            50_000, 60_000, 70_000, 80_000,
        ]


def test_generated_configs_lock_fair_2hop_and_rung_budget():
    files = make_configs.expected_files()
    rows_by_config = {
        str(row["config"]): row for row in make_configs.unique_rows()
    }
    configs = {
        path.name: text
        for path, text in files.items()
        if path.name.startswith("train_")
    }
    assert len(configs) == 21
    for name, text in configs.items():
        row = rows_by_config[name]
        rung = len(row["sources"])
        assert "n_hop: 2\n" in text
        assert 'neighbor_sampling_hop_sizes: "9,9"\n' in text
        assert "neighbor_sampling_node_limit: 101\n" in text
        assert "neighbor_matching_walk_hops: 1\n" in text
        assert "layers: S,U,M\n" in text
        assert "gnn_type: sage\n" in text
        assert "n_layer: 1\n" in text
        assert "dropout: 0\n" in text
        assert "dataset_len_cap: 10000\n" in text
        assert f"epochs: {rung}\n" in text
        assert "checkpoint_step: 10000\n" in text
        assert "neighbor_sampling_episode_source_weighting: balanced\n" in text
        assert "prefix: nm_ladder_fx10k_h2m_" in text


def test_manifest_records_target_step_for_every_row():
    manifest = make_configs.render_manifest(make_configs.plan()).splitlines()
    assert manifest[0].split("\t")[4] == "target_step"
    assert len(manifest) == 25
    assert manifest[1].split("\t")[4] == "10000"
    assert manifest[8].split("\t")[4] == "80000"


def test_smoke_and_eval_use_the_fair_2hop_sampler():
    files = make_configs.expected_files()
    smoke = files[make_configs.CONFIG_DIR / "smoke_election.yaml"]
    assert "dataset: election2020\n" in smoke
    assert "dataset_len_cap: 20\n" in smoke
    assert 'checkpoint_steps: "0,20"\n' in smoke
    assert 'neighbor_sampling_hop_sizes: "9,9"\n' in smoke
    assert "neighbor_sampling_node_limit: 101\n" in smoke
    assert "neighbor_matching_walk_hops: 1\n" in smoke
    assert "prefix: nm_ladder_fx10k_h2m_smoke_election\n" in smoke

    eval_script = (SETUP / "eval_ladder_tucker.sh").read_text(encoding="utf-8")
    assert "--n_hop 2" in eval_script
    assert "--neighbor_sampling_hop_sizes 9,9" in eval_script
    assert "--neighbor_sampling_node_limit 101" in eval_script
    assert "--neighbor_matching_walk_hops 1" in eval_script


def test_checkpoint_requests_are_rung_specific():
    requested = make_model_list.requested_models("A")
    assert [step for _, step in requested] == [
        10_000, 20_000, 30_000, 40_000,
        50_000, 60_000, 70_000, 80_000,
    ]


def test_all_models_resolve_at_their_own_final_step(tmp_path):
    requested = make_model_list.requested_models("all")
    for index, (prefix, step) in enumerate(requested):
        checkpoint_dir = tmp_path / f"{prefix}_{index:02d}" / "checkpoint"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / f"state_dict_{step}.ckpt").touch()

    resolved, missing = make_model_list.resolve_models(tmp_path, "all")
    assert not missing
    assert len(resolved) == 21
    assert [(prefix, step) for prefix, step, _ in resolved] == requested


def test_checkpoint_resolver_skips_newer_incomplete_retry(tmp_path):
    prefix = "nm_ladder_fx10k_h2m_ordA_r3"
    complete = tmp_path / f"{prefix}_02_08_2026_10_00_00" / "checkpoint"
    complete.mkdir(parents=True)
    checkpoint = complete / "state_dict_30000.ckpt"
    checkpoint.touch()

    incomplete = tmp_path / f"{prefix}_02_08_2026_11_00_00"
    incomplete.mkdir()
    incomplete.touch()

    assert make_model_list.complete_checkpoint(tmp_path, prefix, 30_000) == checkpoint
    assert make_model_list.complete_checkpoint(tmp_path, prefix, 40_000) is None

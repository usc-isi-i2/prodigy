from pathlib import Path
import sys

import pytest
import yaml


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from radius_plan import (  # noqa: E402
    ARMS,
    CHECKPOINT_STEPS,
    PANELS,
    select_validation_checkpoint,
)


def test_three_arm_registry_is_exact():
    assert [arm.arm_id for arm in ARMS] == ["global", "radius_mix", "close_only"]
    assert [arm.radii for arm in ARMS] == [
        ("global",),
        ("2", "3", "global"),
        ("2", "3"),
    ]


@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: arm.arm_id)
def test_configs_match_final_core_without_source_confinement(arm):
    config = yaml.safe_load(arm.config.read_text(encoding="utf-8"))
    assert config["graph_filename"] == "ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt"
    assert config["edge_view"] == "static_train"
    assert config["target_edge_view"] == "static_test"
    assert config["neighbor_matching_edge_split"] is True
    assert (config["n_hop"], config["neighbor_sampling_hop_sizes"]) == (2, "9,9")
    assert config["neighbor_sampling_node_limit"] == 101
    assert config["neighbor_matching_walk_hops"] == 1
    assert (config["n_way"], config["n_shots"], config["n_query"]) == (30, 3, 4)
    assert config["batch_size"] == 4
    assert config["dataset_len_cap"] == 2500
    assert config["epochs"] == 1
    assert config["checkpoint_steps"] == "100,300,900,2500"
    assert "neighbor_sampling_episode_source" not in config
    assert "neighbor_sampling_strata" not in config
    assert "neighbor_sampling_source_subset" not in config
    assert tuple(config["neighbor_sampling_center_radii"].split(",")) == arm.radii


def test_validation_panels_and_macro_selection_are_predeclared():
    assert [(panel.panel_id, panel.primary) for panel in PANELS] == [
        ("radius2", True),
        ("radius3", True),
        ("global", True),
        ("within_source", False),
    ]
    rows = []
    for step in CHECKPOINT_STEPS:
        score = 0.8 if step in {300, 900} else 0.2
        for panel in ("radius2", "radius3", "global"):
            rows.append({"checkpoint_step": step, "panel": panel, "score": score})
    selection = select_validation_checkpoint(rows)
    assert selection["selected"]["checkpoint_step"] == 300


def test_selector_accepts_the_10k_convergence_schedule():
    steps = (2500, 5000, 7500, 10000)
    rows = [
        {"checkpoint_step": step, "panel": panel, "score": float(step)}
        for step in steps
        for panel in ("radius2", "radius3", "global")
    ]
    selection = select_validation_checkpoint(rows, checkpoint_steps=steps)
    assert selection["selected"]["checkpoint_step"] == 10000


def test_selector_rejects_missing_panel_cell():
    rows = [
        {"checkpoint_step": step, "panel": panel, "score": 0.5}
        for step in CHECKPOINT_STEPS
        for panel in ("radius2", "radius3", "global")
    ]
    with pytest.raises(ValueError, match="every checkpoint"):
        select_validation_checkpoint(rows[:-1])

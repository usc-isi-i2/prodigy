from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "make_configs.py"
SPEC = importlib.util.spec_from_file_location("seq_make_configs", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_rungs_are_exactly_40k_and_prefix_ordered() -> None:
    rows = MODULE.plan()
    assert len(rows) == 8
    for rung, row in enumerate(rows, 1):
        assert row["rung"] == rung
        assert row["sources"] == [source[0] for source in MODULE.SOURCES[:rung]]
        assert len(row["steps"]) == rung
        assert sum(row["steps"]) == 40_000
        assert max(row["steps"]) - min(row["steps"]) <= 1
        assert row["boundaries"][-1] == 40_000


def test_known_step_allocations() -> None:
    assert MODULE.allocate_steps(1) == [40_000]
    assert MODULE.allocate_steps(2) == [20_000, 20_000]
    assert MODULE.allocate_steps(3) == [13_334, 13_333, 13_333]
    assert MODULE.allocate_steps(8) == [5_000] * 8


def test_rendered_configs_lock_the_fair_two_hop_tuple() -> None:
    for row in MODULE.plan():
        config = MODULE.render_config(row)
        for required in (
            "n_hop: 2",
            'neighbor_sampling_hop_sizes: "9,9"',
            "neighbor_sampling_node_limit: 101",
            "neighbor_matching_walk_hops: 1",
            "layers: S,U,M",
            "gnn_type: sage",
            "epochs: 4",
        ):
            assert required in config
        assert f"prefix: nm_ladder_seq_h2m_r{row['rung']}" in config

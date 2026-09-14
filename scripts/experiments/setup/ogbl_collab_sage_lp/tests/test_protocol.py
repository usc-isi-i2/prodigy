from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "run.py"
SPEC = importlib.util.spec_from_file_location("ogbl_collab_sage_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def test_unique_train_graph_collapses_direction_and_multiplicity() -> None:
    edges = np.array([[0, 1], [1, 0], [0, 1], [1, 2], [3, 3]], dtype=np.int64)
    graph, _ = run.unique_unweighted_train_graph(edges, num_nodes=4)
    assert {tuple(edge) for edge in graph.T.tolist()} == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (3, 3),
    }


def test_unique_train_graph_fingerprint_is_order_invariant() -> None:
    edges = np.array([[0, 1], [1, 2], [0, 1]], dtype=np.int64)
    first, first_hash = run.unique_unweighted_train_graph(edges, num_nodes=3)
    second, second_hash = run.unique_unweighted_train_graph(edges[::-1], num_nodes=3)
    assert np.array_equal(first, second)
    assert first_hash == second_hash

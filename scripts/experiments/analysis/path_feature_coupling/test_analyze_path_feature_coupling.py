from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import shortest_path


MODULE_PATH = Path(__file__).with_name("analyze_path_feature_coupling.py")
SPEC = importlib.util.spec_from_file_location("path_feature_coupling", MODULE_PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_sampled_prefixes_have_exact_shortest_path_lengths() -> None:
    # Cycle plus tails supplies many exact 1/2/3 prefixes and genuinely far pairs.
    edges = []
    for node in range(20):
        edges.append((node, (node + 1) % 20))
    edges.extend((node, node + 20) for node in range(10))
    edge_index = np.asarray(edges, dtype=np.int64).T
    adjacency = MOD.build_undirected_csr(edge_index, 30)
    blocks = MOD.propose_blocks(
        adjacency,
        n_nodes=30,
        n_blocks=20,
        rng=np.random.default_rng(7),
        attempts_per_block=1_000,
    )
    assert len(blocks) == 20
    for block in blocks:
        distances = shortest_path(
            adjacency, directed=False, unweighted=True, indices=block.anchor
        )
        assert distances[block.d1] == 1
        assert distances[block.d2] == 2
        assert distances[block.d3] == 3
        assert distances[block.far] > 3


def test_single_coordinate_probe_finds_sparse_signal() -> None:
    rng = np.random.default_rng(11)
    n_train, n_test, dim = 1_000, 400, 32
    y_train = np.repeat([0, 1], n_train // 2)
    y_test = np.repeat([0, 1], n_test // 2)
    train = rng.normal(size=(n_train, dim)).astype(np.float32)
    test = rng.normal(size=(n_test, dim)).astype(np.float32)
    train[:, 9] += 2.5 * y_train
    test[:, 9] += 2.5 * y_test
    result = MOD.single_coordinate_probe(train, y_train, test, y_test)
    assert result["selected_dimension"] == 9
    assert result["test_auc_oriented"] > 0.9

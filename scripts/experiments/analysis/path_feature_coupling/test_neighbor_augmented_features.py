import numpy as np
from scipy.sparse import csr_matrix

from analyze_neighbor_augmented_features import (
    distance_summary,
    identity_probe,
    sampled_neighbor_means,
    spaces,
)


def test_sampled_neighbor_means_include_zero_rows_and_exclude_center():
    # 0--1--2. Node 1 has a deliberately missing/zero feature row.
    adjacency = csr_matrix(
        (
            np.ones(4, dtype=np.uint8),
            (np.array([0, 1, 1, 2]), np.array([1, 0, 2, 1])),
        ),
        shape=(3, 3),
    )
    x = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    means, degree, count = sampled_neighbor_means(
        x, adjacency, np.array([0, 1, 2]), fanout=10, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(means, [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]])
    np.testing.assert_array_equal(degree, [1, 2, 1])
    np.testing.assert_array_equal(count, [1, 2, 1])


def test_spaces_concatenate_center_and_neighbor_mean():
    raw = np.array([[1.0, 2.0]], dtype=np.float32)
    mean = np.array([[3.0, 4.0]], dtype=np.float32)
    result = spaces(raw, mean)
    np.testing.assert_allclose(result["raw_center"], raw)
    np.testing.assert_allclose(result["neighbor_mean"], mean)
    np.testing.assert_allclose(result["center_plus_neighbor_mean"], [[1, 2, 3, 4]])


def test_distance_summary_uses_fixed_pair_indices():
    rows = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = distance_summary(rows, rows, np.array([0]), np.array([1]))
    assert out["n"] == 1
    assert np.isclose(out["mean_cosine_distance"], 1.0)
    assert np.isclose(out["mean_euclidean_distance"], np.sqrt(2.0))


def test_identity_probe_skips_single_graph_pilots():
    one = np.ones((4, 2), dtype=np.float32)
    samples = {"g": spaces(one, one)}
    result = identity_probe(samples, ["g"], "raw_center", seed=0)
    assert "at least two graphs" in result["error"]

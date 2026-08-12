import numpy as np
from scipy.sparse import csr_matrix

from analyze_neighbor_augmented_features import (
    distance_summary,
    identity_probe,
    lda_projection,
    projection_split,
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


def test_projection_split_is_balanced_and_deterministic():
    one = np.ones((10, 2), dtype=np.float32)
    samples = {name: spaces(one, one) for name in ("a", "b", "c")}
    mask1, labels1, metadata1 = projection_split(samples, ["a", "b", "c"], seed=7)
    mask2, labels2, metadata2 = projection_split(samples, ["a", "b", "c"], seed=7)
    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_array_equal(labels1, labels2)
    assert metadata1 == metadata2
    assert metadata1["n_test"] == 9
    assert all(v["test"] == 3 for v in metadata1["counts_by_graph"].values())


def test_lda_projection_is_fit_on_train_and_separates_heldout_classes():
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(3), 20)
    matrix = np.vstack(
        [rng.normal(loc=label * 5.0, scale=0.2, size=(20, 4)) for label in range(3)]
    ).astype(np.float32)
    test_mask = np.zeros(60, dtype=bool)
    for label in range(3):
        test_mask[label * 20 : label * 20 + 5] = True
    coordinates, metadata = lda_projection(matrix, labels, ~test_mask)
    assert coordinates.shape == (60, 3)
    assert metadata["heldout_nearest_centroid_balanced_accuracy_in_3d"] == 1.0


def test_lda_projection_skips_single_graph_pilot():
    matrix = np.ones((10, 4), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int16)
    train_mask = np.arange(10) >= 3
    coordinates, metadata = lda_projection(matrix, labels, train_mask)
    np.testing.assert_array_equal(coordinates, np.zeros((10, 3), dtype=np.float32))
    assert "at least two graphs" in metadata["error"]

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "run.py"
SPEC = importlib.util.spec_from_file_location("ogbl_collab_mlp_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class FakeEvaluator:
    K = 50

    def eval(self, values):
        positive = values["y_pred_pos"]
        negative = values["y_pred_neg"]
        threshold = np.sort(negative)[-self.K] if len(negative) >= self.K else -np.inf
        return {f"hits@{self.K}": float(np.mean(positive > threshold))}


def test_pair_keys_are_undirected() -> None:
    edges = np.array([[1, 3], [3, 1], [0, 2]], dtype=np.int64)
    keys = run.pair_keys(edges, 5)
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


def test_test_strata_distinguish_training_and_validation_history() -> None:
    train = np.array([[0, 1], [1, 2]], dtype=np.int64)
    valid = np.array([[2, 3]], dtype=np.int64)
    test = np.array([[1, 0], [3, 2], [0, 3]], dtype=np.int64)
    strata = run.test_strata(train, valid, test, 4)
    assert strata["seen_in_train"].tolist() == [True, False, False]
    assert strata["novel_vs_train"].tolist() == [False, True, True]
    assert strata["seen_pre2019"].tolist() == [True, True, False]
    assert strata["novel_vs_pre2019"].tolist() == [False, False, True]


def test_training_stream_is_repeatable_and_arm_independent() -> None:
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    first = run.epoch_training_pairs(edges, 5, seed=7, epoch=3)
    second = run.epoch_training_pairs(edges, 5, seed=7, epoch=3)
    different_epoch = run.epoch_training_pairs(edges, 5, seed=7, epoch=4)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[2] != different_epoch[2]


def test_encoder_shapes_and_cosine_symmetry() -> None:
    features = torch.randn(6, 128)
    edges = torch.tensor([[0, 1], [2, 3], [4, 5]])
    for arm in run.LEARNED_ARMS:
        model = run.Encoder(arm)
        forward = model.logits(features[edges[:, 0]], features[edges[:, 1]])
        reverse = model.logits(features[edges[:, 1]], features[edges[:, 0]])
        assert forward.shape == (3,)
        assert torch.allclose(forward, reverse)


def test_ranking_metrics_include_official_and_secondary_views() -> None:
    positive = np.linspace(0.5, 1.0, 60)
    negative = np.linspace(0.0, 0.49, 100)
    metrics = run.ranking_metrics(FakeEvaluator(), positive, negative)
    assert metrics["hits_at_10"] == 1.0
    assert metrics["hits_at_50"] == 1.0
    assert metrics["hits_at_100"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0

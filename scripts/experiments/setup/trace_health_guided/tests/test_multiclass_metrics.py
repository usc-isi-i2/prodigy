import numpy as np
import pytest
from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics, paired_summary


def test_local_class_permutation_preserves_metrics():
    y = np.tile(np.arange(3), 3)
    z = np.eye(3)[y] * 3
    permutation = np.array([2, 0, 1])
    inverse = np.argsort(permutation)
    assert episode_metrics(y, z) == episode_metrics(inverse[y], z[:, permutation])


def test_ranking_can_be_perfect_despite_prediction_collapse():
    y = np.tile(np.arange(3), 3)
    z = np.eye(3)[y] * 2
    z[:, 0] += 10
    scores = episode_metrics(y, z)
    assert scores['ovr_auc'] == 1
    assert scores['accuracy'] == pytest.approx(1 / 3)
    assert scores['macro_f1'] < .2


def test_paired_identity_and_missing_class_guards():
    y = np.tile(np.arange(3), 2)
    z = np.eye(3)[y]
    a = {'episode': (y, z)}
    assert paired_summary(a, a, draws=10)['accuracy']['interval'] == [0, 0]
    with pytest.raises(ValueError):
        paired_summary(a, {'episode': (y[::-1], z)})
    with pytest.raises(ValueError):
        episode_metrics(np.zeros(6, dtype=int), z)

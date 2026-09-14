import pytest

from mixture_scaling.probe_strict import choose_trial


TRIALS = [
    {"c": 0.1, "roc_auc_ovr_macro": 0.8, "f1_macro": 0.6},
    {"c": 1.0, "roc_auc_ovr_macro": 0.7, "f1_macro": 0.9},
]


def test_choose_trial_uses_requested_metric():
    assert choose_trial(TRIALS, "auc")["c"] == 0.1
    assert choose_trial(TRIALS, "f1")["c"] == 1.0


def test_choose_trial_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown selection metric"):
        choose_trial(TRIALS, "accuracy")

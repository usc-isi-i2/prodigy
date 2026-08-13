from experiments.params import get_params
from experiments.trainer import TrainerFS
from unittest.mock import patch


def test_eval_only_split_can_lock_validation_only():
    params = get_params(["--eval_only", "True", "--eval_only_split", "val"])
    assert params["eval_only"] is True
    assert params["eval_only_split"] == "val"


def test_eval_only_split_preserves_test_default():
    params = get_params(["--eval_only", "True"])
    assert params["eval_only_split"] == "test"


def _eval_only_trainer(split):
    trainer = TrainerFS.__new__(TrainerFS)
    trainer.parameter = {
        "eval_only": True,
        "eval_only_split": split,
        "eval_test_before_train": False,
        "eval_val_before_train": False,
    }
    trainer.val_dataloader = object()
    trainer.test_dataloader = object()
    trainer.is_regression = False
    trainer.is_feature_prediction = False
    calls = []

    def do_eval(dataloader, split_name, step):
        calls.append((dataloader, split_name, step))
        return 1.0, 0.5, 0.01, 0.0, None

    trainer.do_eval = do_eval
    return trainer, calls


def test_validation_only_train_path_never_calls_test():
    trainer, calls = _eval_only_trainer("val")
    with patch("experiments.trainer.wandb"):
        trainer.train()
    assert [(split_name, step) for _, split_name, step in calls] == [("val", 0)]


def test_test_only_train_path_never_calls_validation():
    trainer, calls = _eval_only_trainer("test")
    with patch("experiments.trainer.wandb"):
        trainer.train()
    assert [(split_name, step) for _, split_name, step in calls] == [("test", 0)]

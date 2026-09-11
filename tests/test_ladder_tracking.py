import json
from types import SimpleNamespace

import pytest

from mixture_scaling.ladder_tracking import LossWindow, tracked_run


def test_window_weights_short_batches_and_resets_source_membership():
    window = LossWindow()
    window.add("a", 2.0, 3)
    window.add("b", 4.0, 1)
    metrics = window.flush()
    assert metrics["train/loss"] == 2.5
    assert metrics["train/source/a/loss"] == 2
    assert metrics["train/source/b/loss"] == 4
    window.add("a", 7, 1)
    assert "train/source/b/loss" not in window.flush()
    with pytest.raises(ValueError, match="empty"):
        window.flush()


def test_offline_run_writes_syncable_history_without_login(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    args = SimpleNamespace(wandb_project="ladder-tracking-test", wandb_mode="offline",
                           wandb_group="test", log_interval=25)
    with tracked_run(tmp_path, {"run_id": "test-rung", "sources": ["a"]}, args) as (run, log):
        assert run.offline
        log(25, {"train/loss": 0.5})
        log(50, {"train/loss": 0.4})
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["mode"] == "offline"
    assert len(list(tmp_path.glob("wandb/offline-run-*/run-*.wandb"))) == 1
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in rows] == [25, 50]
    assert rows[-1]["train/loss"] == 0.4

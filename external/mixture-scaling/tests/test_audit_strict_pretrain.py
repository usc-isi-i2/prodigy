import json

import pytest
import torch

from mixture_scaling.audit_strict_pretrain import audit_run


def write_run(path, *, checkpoint_step=20):
    path.mkdir()
    summary = {
        "status": "complete", "sources": ["cora"], "seed": 0,
        "source_confined": True, "split_hashes": {"cora": "hash"},
        "ssl_train_partition": "train_nodes/train_edges",
        "ssl_validation_partition": "train_nodes/heldout_edges",
        "ssl_edge_validation_fraction": 0.1,
        "best_step": 20, "best_validation_loss": 0.4,
    }
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "validation.jsonl").write_text(
        json.dumps({"step": 10, "validation_loss": 0.5}) + "\n" +
        json.dumps({"step": 20, "validation_loss": 0.4}) + "\n"
    )
    torch.save({"step": checkpoint_step}, path / "best.pt")


def test_audit_accepts_absolute_best(tmp_path):
    run = tmp_path / "run"
    write_run(run)
    assert audit_run(run, ["cora"], 0)["best_step"] == 20


def test_audit_rejects_wrong_checkpoint(tmp_path):
    run = tmp_path / "run"
    write_run(run, checkpoint_step=10)
    with pytest.raises(ValueError, match="best_step"):
        audit_run(run, ["cora"], 0)

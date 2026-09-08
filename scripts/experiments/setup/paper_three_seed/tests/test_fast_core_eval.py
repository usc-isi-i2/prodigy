import json
from pathlib import Path

import pytest

from scripts.experiments.setup.paper_three_seed.evaluate_fast_core import (
    GROUP_COUNTS,
    audit_cells,
    parse_run_group,
    plan_registry,
    repo_relative_config,
    validate_terminal_status,
)


def test_repo_relative_config_survives_different_worktree_roots():
    value = "/dataMeR1/phil/gfm/other-worktree/scripts/experiments/setup/example/train.yaml"
    assert repo_relative_config(value) == "scripts/experiments/setup/example/train.yaml"


def test_group_registries_match_the_registered_plan():
    assert len(plan_registry("onehop")) == 18
    assert len(plan_registry("twohop")) == 23


def test_parse_run_group_rejects_unknown_kind(tmp_path):
    assert parse_run_group(f"onehop={tmp_path}") == ("onehop", tmp_path.resolve())
    with pytest.raises(Exception):
        parse_run_group(f"other={tmp_path}")


def test_terminal_status_fails_closed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(
        json.dumps({"status": "failed_or_interrupted", "finished": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        validate_terminal_status(run, GROUP_COUNTS["onehop"])


def test_audit_requires_one_fingerprint_per_target(tmp_path, monkeypatch):
    from scripts.experiments.setup.paper_three_seed import evaluate_fast_core as module

    monkeypatch.setattr(module, "TARGETS", ("a", "b"))
    jobs = [
        {"model_id": "m1", "seed": 1, "family": "ladder_1hop"},
        {"model_id": "m2", "seed": 2, "family": "ladder_1hop"},
    ]
    for model in ("m1", "m2"):
        for target in ("a", "b"):
            path = tmp_path / "cells" / model / f"{target}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "protocol": module.PROTOCOL,
                        "model_id": model,
                        "target": target,
                        "episodes": 512,
                        "fingerprint": f"fixed-{target}",
                        "roc_auc": 0.6,
                        "accuracy": 0.5,
                        "loss": 1.0,
                    }
                ),
                encoding="utf-8",
            )
    audit = audit_cells(tmp_path, jobs, 512)
    assert audit["cells"] == 4
    assert audit["training_seeds"] == [1, 2]
    assert audit["fingerprints"] == {"a": "fixed-a", "b": "fixed-b"}

import json
import os
from pathlib import Path

import pytest

from experiments.run_shared_graph import prepare_interrupted_recovery


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def job(index: int) -> dict:
    return {
        "config": f"config_{index}.yaml",
        "prefix": f"model_{index}",
        "seed": index,
        "workers": 4,
        "device": f"cuda:{index % 2}",
        "exp_name": f"model_{index}_original",
        "timestamp": "original-attempt",
    }


def test_recovery_reuses_terminal_and_archives_only_interrupted(tmp_path: Path) -> None:
    jobs = [job(0), job(1)]
    write_json(tmp_path / "manifest.json", {"revision": "old", "jobs": jobs})

    checkpoint_dir = tmp_path / "state" / "complete" / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "state_dict_2500.ckpt").write_bytes(b"weights")
    write_json(
        tmp_path / "job_000" / "result.json",
        {"status": "complete", "checkpoint_dir": str(checkpoint_dir)},
    )
    write_json(tmp_path / "job_001" / "result.json", {"status": "training", "pid": 999_999_999})
    (tmp_path / "job_001" / "console.log").write_text("partial", encoding="utf-8")

    requested = [
        dict(row, device="cuda:3", exp_name="fresh", timestamp="recovery-attempt")
        for row in jobs
    ]
    plan, pending, reused = prepare_interrupted_recovery(tmp_path, requested, "recovery-rev")

    assert reused == [{"job": 0, "exitcode": 0, "reused_terminal": True}]
    assert [index for index, _ in pending] == [1]
    assert pending[0][1]["workers"] == 4
    assert pending[0][1]["exp_name"].startswith("model_1_original_recovery_")
    assert (tmp_path / "job_000" / "result.json").is_file()
    assert not (tmp_path / "job_001").exists()
    history = Path(plan["recoveries"][-1]["history"])
    assert (history / "job_001" / "console.log").read_text(encoding="utf-8") == "partial"
    assert plan["revision"] == "old"
    assert plan["recoveries"][-1]["launcher_revision"] == "recovery-rev"


def test_recovery_rejects_parameter_drift_before_mutation(tmp_path: Path) -> None:
    original = job(0)
    write_json(tmp_path / "manifest.json", {"jobs": [original]})
    requested = [dict(original, workers=0)]

    with pytest.raises(ValueError, match="workers"):
        prepare_interrupted_recovery(tmp_path, requested, "new")

    assert not (tmp_path / "history").exists()


def test_recovery_rejects_live_recorded_trainer(tmp_path: Path) -> None:
    original = job(0)
    write_json(tmp_path / "manifest.json", {"jobs": [original]})
    write_json(tmp_path / "job_000" / "result.json", {"status": "training", "pid": os.getpid()})

    with pytest.raises(RuntimeError, match="is alive"):
        prepare_interrupted_recovery(tmp_path, [original], "new")

    assert (tmp_path / "job_000" / "result.json").is_file()

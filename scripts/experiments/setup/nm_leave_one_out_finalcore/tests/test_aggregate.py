from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SETUP = HERE.parent
FINAL_CORE = SETUP.parent / "final_core"
sys.path.insert(0, str(FINAL_CORE))
sys.path.insert(0, str(SETUP))

from aggregate_results import aggregate
from auc_contract import METRIC_CONTRACT
from loo_plan import CHECKPOINT_STEP, EPISODE_COUNT, PROTOCOL, checkpoint_path, physical_jobs


def test_complete_heldout_sweep_aggregates(tmp_path):
    training = tmp_path / "training"
    results = tmp_path / "results"
    output = tmp_path / "summary"
    manifest_jobs = []
    for index, job in enumerate(physical_jobs()):
        manifest_jobs.append({
            "prefix": job.model.model_id,
            "exp_name": f"run_{index:03d}",
        })
    training.mkdir()
    (training / "manifest.json").write_text(
        json.dumps({"jobs": manifest_jobs}), encoding="utf-8"
    )
    for target_index, job in enumerate(physical_jobs(), 1):
        target = job.model.heldout
        value = 0.5 + target_index / 100
        fingerprint = (str(target_index) * 64)[:64]
        payload = {
            "protocol": PROTOCOL,
            "metric_contract": METRIC_CONTRACT,
            "checkpoint_step": CHECKPOINT_STEP,
            "checkpoint": str(checkpoint_path(training, job, "")),
            "split": "test",
            "edge_view": "static_train",
            "target_edge_view": "static_test",
            "batch_size": 32,
            "batch_count": 16,
            "episode_count": EPISODE_COUNT,
            "seed": 0,
            "model_id": job.model.model_id,
            "sources": list(job.model.sources),
            "target": target,
            "score": value,
            "accuracy": value,
            "f1_macro": value,
            "roc_auc_ovr_macro": value,
            "score_std": 0.01,
            "loss": 1.0,
            "aux_loss": 0.0,
            "episode_plan_fingerprint": fingerprint,
            "observed_episode_fingerprint": fingerprint,
        }
        path = results / "seed_0" / job.model.model_id / f"{target}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    aggregate(results, output, training, 32)
    with (output / "loo_heldout_metrics.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 9
    assert [row["heldout"] for row in rows] == [job.model.heldout for job in physical_jobs()]
    receipt = json.loads((output / "completeness.json").read_text(encoding="utf-8"))
    assert receipt["training_seeds"] == [0]
    assert receipt["loo_models"] == 9
    assert receipt["heldout_test_cells"] == 9

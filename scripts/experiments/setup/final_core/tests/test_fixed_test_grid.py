import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aggregate_fixed_test import aggregate, load_and_validate  # noqa: E402
from core_plan import SOURCES  # noqa: E402
from fixed_test_plan import (  # noqa: E402
    CHECKPOINT_STEP,
    EPISODE_COUNT,
    PROTOCOL,
    expected_counts,
    model_for_ladder,
    physical_jobs,
)


def test_exact_combined_matrix_ladder_counts():
    assert expected_counts() == {
        "matrix_checkpoint_seeds": 27,
        "ladder_checkpoint_seeds": 75,
        "overlap_checkpoint_seeds": 9,
        "union_checkpoint_seeds": 93,
        "matrix_cells": 243,
        "ladder_physical_cells": 675,
        "overlap_cells": 81,
        "union_cells": 837,
        "ladder_reported_rows": 729,
    }
    assert len(physical_jobs()) == 93
    assert model_for_ladder("A", 9).model_id == "all9"
    assert model_for_ladder("B", 9).model_id == "all9"
    assert model_for_ladder("C", 9).model_id == "all9"


def write_fake_grid(root: Path, batch_size: int = 64) -> None:
    plan_fingerprints = {
        target: f"{index + 1:064x}" for index, target in enumerate(SOURCES)
    }
    observed_fingerprints = {
        target: f"{index + 101:064x}" for index, target in enumerate(SOURCES)
    }
    for job_index, job in enumerate(physical_jobs()):
        for target_index, target in enumerate(SOURCES):
            path = root / f"seed_{job.seed}" / job.model.model_id / f"{target}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            score = 0.1 + ((job_index * len(SOURCES) + target_index) % 700) / 1000
            payload = {
                "protocol": PROTOCOL,
                "created_utc": "2026-08-08T00:00:00+00:00",
                "evaluation_commit": "a" * 40,
                "worker_index": job_index % 8,
                "model_id": job.model.model_id,
                "aliases": list(job.model.aliases),
                "sources": list(job.model.sources),
                "seed": job.seed,
                "target": target,
                "checkpoint_step": CHECKPOINT_STEP,
                "checkpoint": (
                    f"/state/finalcore_{job.model.model_id}_s{job.seed}/checkpoint/"
                    f"state_dict_{CHECKPOINT_STEP}.ckpt"
                ),
                "split": "test",
                "edge_view": "static_train",
                "target_edge_view": "static_test",
                "batch_size": batch_size,
                "batch_count": EPISODE_COUNT // batch_size,
                "episode_count": EPISODE_COUNT,
                "episode_plan_fingerprint": plan_fingerprints[target],
                "observed_episode_fingerprint": observed_fingerprints[target],
                "elapsed_seconds": 1.0,
                "max_rss_gib": 118.0,
                "peak_cuda_allocated_gib": 10.0,
                "score": score,
                "score_std": 0.01,
                "loss": 1.0,
                "aux_loss": 0.0,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")


def test_strict_aggregate_outputs_exact_row_counts(tmp_path):
    results = tmp_path / "results"
    output = tmp_path / "summary"
    write_fake_grid(results)
    aggregate(results, output, expected_batch_size=64)
    assert len((output / "combined_physical_cells.tsv").read_text().splitlines()) - 1 == 837
    assert len((output / "single_source_matrix_long.tsv").read_text().splitlines()) - 1 == 243
    assert len((output / "ladder_physical_cells.tsv").read_text().splitlines()) - 1 == 675
    assert len((output / "ladder_results_alias_expanded.tsv").read_text().splitlines()) - 1 == 729
    assert len((output / "matrix_ladder_rung1_overlap.tsv").read_text().splitlines()) - 1 == 81
    assert len((output / "single_source_matrix_seed_0.csv").read_text().splitlines()) - 1 == 9
    completeness = json.loads((output / "completeness.json").read_text())
    assert completeness["union_cells"] == 837
    assert completeness["batch_size"] == 64
    assert completeness["batch_count"] == 8


def test_fingerprint_disagreement_fails(tmp_path):
    results = tmp_path / "results"
    write_fake_grid(results)
    path = results / "seed_0" / "ss_ukr_rus" / "ukr_rus.json"
    payload = json.loads(path.read_text())
    payload["observed_episode_fingerprint"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_and_validate(results, expected_batch_size=64)
    except ValueError as error:
        assert "distinct observed_episode_fingerprint" in str(error)
    else:
        raise AssertionError("fingerprint mismatch must fail")


def test_validation_or_selection_metadata_is_forbidden(tmp_path):
    results = tmp_path / "results"
    write_fake_grid(results)
    path = results / "seed_0" / "ss_ukr_rus" / "ukr_rus.json"
    payload = json.loads(path.read_text())
    payload["selected_checkpoint_step"] = 2500
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_and_validate(results, expected_batch_size=64)
    except ValueError as error:
        assert "selection fields are forbidden" in str(error)
    else:
        raise AssertionError("selection metadata must fail")

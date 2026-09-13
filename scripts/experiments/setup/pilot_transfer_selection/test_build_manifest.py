import csv
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_manifest import (  # noqa: E402
    PILOT_STEPS,
    SEEDS,
    SOURCES,
    build_rows,
    checkpoint_jobs,
    validate_rows,
    write_tsv,
)


def test_stage_a_manifest_is_exact() -> None:
    rows = build_rows(state_root=Path("/frozen/state"), run_stamp="stamp")
    assert len(rows) == 729
    assert len({row["checkpoint_job_id"] for row in rows}) == 81
    assert len({row["cell_id"] for row in rows}) == 729
    assert {row["checkpoint_step"] for row in rows} == set(PILOT_STEPS)
    assert {row["training_seed"] for row in rows} == set(SEEDS)
    assert {row["target"] for row in rows} == set(SOURCES)
    assert all(row["evaluation_split"] == "validation" for row in rows)
    assert all(row["target_edge_view"] == "static_validation" for row in rows)
    assert all("state_dict_2500.ckpt" not in row["checkpoint"] for row in rows)


def test_duplicate_cell_is_rejected() -> None:
    rows = build_rows(state_root=Path("/frozen/state"), run_stamp="stamp")
    rows[-1] = dict(rows[0])
    try:
        validate_rows(rows)
    except ValueError as error:
        assert "duplicate cell IDs" in str(error)
    else:
        raise AssertionError("duplicate manifest cell must fail")


def test_tsv_round_trip(tmp_path: Path) -> None:
    rows = build_rows(state_root=Path("/frozen/state"), run_stamp="stamp")
    output = tmp_path / "manifest.tsv"
    write_tsv(rows, output)
    with output.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle, delimiter="\t"))
    assert len(loaded) == 729
    assert loaded[0]["protocol_id"] == "pilot_transfer_selection_v1"
    assert loaded[-1]["target"] == SOURCES[-1]


def test_checkpoint_job_manifests_are_exact() -> None:
    rows = build_rows(state_root=Path("/frozen/state"), run_stamp="stamp")
    for step in PILOT_STEPS:
        jobs = checkpoint_jobs(rows, step)
        assert len(jobs) == 27
        assert len({(job["model_id"], job["seed"]) for job in jobs}) == 27
        assert all(Path(job["checkpoint"]).name == f"state_dict_{step}.ckpt" for job in jobs)
        assert all(len(job["sources"]) == 1 for job in jobs)

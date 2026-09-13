import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aggregate_stage_a import aggregate, load_and_validate  # noqa: E402
from build_manifest import build_rows, write_tsv  # noqa: E402


def write_fake_grid(root: Path, manifest: Path, batch_size: int = 32) -> None:
    rows = build_rows(state_root=Path("/frozen/state"), run_stamp="stamp")
    write_tsv(rows, manifest)
    targets = list(dict.fromkeys(row["target"] for row in rows))
    plan = {target: f"{index + 1:064x}" for index, target in enumerate(targets)}
    observed = {
        target: f"{index + 101:064x}" for index, target in enumerate(targets)
    }
    for index, row in enumerate(rows):
        path = (
            root
            / f"step_{row['checkpoint_step']}"
            / f"seed_{row['training_seed']}"
            / row["model_id"]
            / f"{row['target']}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        accuracy = 0.1 + (index % 700) / 1000
        payload = {
            "protocol": "pilot_transfer_selection_v1",
            "metric_contract": "accuracy_f1_macro_roc_auc_ovr_macro_v1",
            "model_id": row["model_id"],
            "sources": [row["sources"]],
            "seed": int(row["training_seed"]),
            "target": row["target"],
            "checkpoint_step": int(row["checkpoint_step"]),
            "checkpoint": row["checkpoint"],
            "split": "val",
            "edge_view": "static_train",
            "target_edge_view": "static_validation",
            "batch_size": batch_size,
            "batch_count": 512 // batch_size,
            "episode_count": 512,
            "episode_plan_fingerprint": plan[row["target"]],
            "observed_episode_fingerprint": observed[row["target"]],
            "score": accuracy,
            "score_std": 0.01,
            "loss": 1.0,
            "aux_loss": 0.0,
            "accuracy": accuracy,
            "f1_macro": accuracy - 0.01,
            "roc_auc_ovr_macro": min(0.99, accuracy + 0.2),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_complete_stage_a_grid_aggregates(tmp_path: Path) -> None:
    results = tmp_path / "results"
    manifest = tmp_path / "manifest.tsv"
    output = tmp_path / "summary"
    write_fake_grid(results, manifest)
    aggregate(results, manifest, output, 32)
    assert len((output / "pilot_metrics_long.tsv").read_text().splitlines()) - 1 == 729
    assert (output / "pilot_accuracy_step300_three_seed_mean.csv").is_file()
    completeness = json.loads((output / "completeness.json").read_text())
    assert completeness["cells"] == 729
    assert completeness["split"] == "val"


def test_split_mismatch_is_rejected(tmp_path: Path) -> None:
    results = tmp_path / "results"
    manifest = tmp_path / "manifest.tsv"
    write_fake_grid(results, manifest)
    path = next(results.glob("step_*/seed_*/*/*.json"))
    payload = json.loads(path.read_text())
    payload["split"] = "test"
    path.write_text(json.dumps(payload))
    try:
        load_and_validate(results, manifest, 32)
    except ValueError as error:
        assert "split expected 'val'" in str(error)
    else:
        raise AssertionError("test result must not enter the validation grid")


def test_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    results = tmp_path / "results"
    manifest = tmp_path / "manifest.tsv"
    write_fake_grid(results, manifest)
    path = next(results.glob("step_300/seed_*/*/covid.json"))
    payload = json.loads(path.read_text())
    payload["observed_episode_fingerprint"] = "f" * 64
    path.write_text(json.dumps(payload))
    try:
        load_and_validate(results, manifest, 32)
    except ValueError as error:
        assert "disagrees on observed_episode_fingerprint" in str(error)
    else:
        raise AssertionError("fingerprint mismatch must fail")

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aggregate_auc_matrix import aggregate, specialist_jobs  # noqa: E402
from auc_contract import (  # noqa: E402
    METRIC_CONTRACT,
    load_metric_sidecar,
    load_reference_fingerprints,
)
from core_plan import SOURCES  # noqa: E402
from fixed_test_plan import CHECKPOINT_STEP, EPISODE_COUNT, PROTOCOL  # noqa: E402


def write_fake_auc_grid(root: Path, batch_size: int = 32) -> None:
    plan = {target: f"{index + 1:064x}" for index, target in enumerate(SOURCES)}
    observed = {target: f"{index + 101:064x}" for index, target in enumerate(SOURCES)}
    for job_index, job in enumerate(specialist_jobs()):
        for target_index, target in enumerate(SOURCES):
            accuracy = 0.1 + ((job_index * 9 + target_index) % 700) / 1000
            path = root / f"seed_{job.seed}" / job.model.model_id / f"{target}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "protocol": PROTOCOL,
                "metric_contract": METRIC_CONTRACT,
                "model_id": job.model.model_id,
                "seed": job.seed,
                "target": target,
                "checkpoint_step": CHECKPOINT_STEP,
                "checkpoint": f"/state/checkpoint/state_dict_{CHECKPOINT_STEP}.ckpt",
                "split": "test",
                "edge_view": "static_train",
                "target_edge_view": "static_test",
                "batch_size": batch_size,
                "batch_count": EPISODE_COUNT // batch_size,
                "episode_count": EPISODE_COUNT,
                "episode_plan_fingerprint": plan[target],
                "observed_episode_fingerprint": observed[target],
                "score": accuracy,
                "score_std": 0.01,
                "loss": 1.0,
                "aux_loss": 0.0,
                "accuracy": accuracy,
                "f1_macro": accuracy - 0.01,
                "roc_auc_ovr_macro": min(0.99, accuracy + 0.2),
            }
            path.write_text(json.dumps(payload), encoding="utf-8")


def test_metric_sidecar_requires_all_three_metrics(tmp_path):
    target = "covid"
    path = tmp_path / f"metrics_test_{target}_step{CHECKPOINT_STEP}.json"
    path.write_text(json.dumps({
        f"test_{target}_accuracy": 0.3,
        f"test_{target}_f1": 0.2,
        f"test_{target}_roc_auc": 0.8,
    }), encoding="utf-8")
    assert load_metric_sidecar(tmp_path, target, CHECKPOINT_STEP) == {
        "accuracy": 0.3,
        "f1_macro": 0.2,
        "roc_auc_ovr_macro": 0.8,
    }
    payload = json.loads(path.read_text())
    del payload[f"test_{target}_roc_auc"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_metric_sidecar(tmp_path, target, CHECKPOINT_STEP)
    except ValueError as error:
        assert "roc_auc" in str(error)
    else:
        raise AssertionError("missing ROC-AUC must fail")


def test_auc_aggregate_is_exactly_three_seed_9x9(tmp_path):
    results = tmp_path / "results"
    summary = tmp_path / "summary"
    write_fake_auc_grid(results)
    aggregate(results, summary, expected_batch_size=32)
    long_rows = (summary / "single_source_metrics_long.tsv").read_text().splitlines()
    assert len(long_rows) - 1 == 243
    for metric in ("accuracy", "f1_macro", "roc_auc_ovr_macro"):
        for seed in (0, 1, 2):
            path = summary / f"single_source_{metric}_seed_{seed}.csv"
            assert len(path.read_text().splitlines()) - 1 == 9
        assert (summary / f"single_source_{metric}_three_seed_mean.csv").is_file()
        assert (summary / f"single_source_{metric}_three_seed_sample_std.csv").is_file()
    completeness = json.loads((summary / "completeness.json").read_text())
    assert completeness["specialist_cells"] == 243
    assert completeness["metric_contract"] == METRIC_CONTRACT


def test_reference_fingerprint_ledger_is_exact(tmp_path):
    path = tmp_path / "episode_fingerprints.tsv"
    rows = ["target\tcell_count\tepisode_count_per_cell\tepisode_plan_fingerprint\tobserved_episode_fingerprint"]
    rows.extend(
        f"{target}\t93\t512\t{index + 1:064x}\t{index + 101:064x}"
        for index, target in enumerate(SOURCES)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_reference_fingerprints(path, tuple(SOURCES))
    assert set(loaded) == set(SOURCES)
    rows.pop()
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    try:
        load_reference_fingerprints(path, tuple(SOURCES))
    except ValueError as error:
        assert "wrong targets" in str(error)
    else:
        raise AssertionError("incomplete fingerprint ledger must fail")

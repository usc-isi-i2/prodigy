"""Tests for the two crossed ladder evaluation plans."""

import json
from types import SimpleNamespace

from scripts.experiments.setup.final_core.core_plan import ORDERS, SOURCES
from scripts.experiments.setup.ladder_cross_task_eval import aggregate, evaluate_nm100


def test_nm100_plan_has_25_unique_seed_zero_checkpoints(tmp_path):
    jobs = evaluate_nm100.ladder_jobs()
    assert len(jobs) == 25
    assert {job.seed for job in jobs} == {0}
    assert {job.model.model_id for job in jobs} == set(aggregate.ladder_models())
    assert (
        evaluate_nm100.checkpoint_path(tmp_path, jobs[0], "stamp").name
        == "state_dict_100.ckpt"
    )


def test_aggregate_accepts_complete_physical_grids_and_expands_shared_rungs(tmp_path):
    nm_root = tmp_path / "nm"
    downstream_paths = [tmp_path / "worker0.jsonl", tmp_path / "worker1.jsonl"]
    models = aggregate.ladder_models()
    for model_id in models:
        for target in SOURCES:
            path = nm_root / "seed_0" / model_id / f"{target}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "protocol": evaluate_nm100.PROTOCOL,
                "checkpoint_step": 100,
                "seed": 0,
                "model_id": model_id,
                "target": target,
                "episode_plan_fingerprint": f"plan-{target}",
                "observed_episode_fingerprint": f"observed-{target}",
                "accuracy": 0.5,
                "f1_macro": 0.5,
                "roc_auc_ovr_macro": 0.5,
            }))

    handles = [path.open("w", encoding="utf-8") for path in downstream_paths]
    try:
        index = 0
        for seed in (0, 1, 2):
            for model_id in models:
                for target in aggregate.DOWNSTREAM_TARGETS:
                    handles[index % 2].write(json.dumps({
                        "training_seed": seed,
                        "model_id": model_id,
                        "dataset": target,
                        "checkpoint_step": 2500,
                        "episode_fingerprint": f"fixed-{target}",
                        "accuracy": 0.5,
                        "f1": 0.5,
                        "roc_auc": 0.5,
                    }) + "\n")
                    index += 1
    finally:
        for handle in handles:
            handle.close()

    nm = aggregate.load_nm(nm_root)
    downstream = aggregate.load_downstream(downstream_paths)
    assert len(nm) == 25 * 9
    assert len(downstream) == 3 * 25 * 4

    nm_csv = tmp_path / "nm.csv"
    downstream_csv = tmp_path / "downstream.csv"
    aggregate.write_ladder_csv(nm_csv, task="neighbor_matching", lookup=nm)
    aggregate.write_ladder_csv(downstream_csv, task="classification", lookup=downstream)
    assert len(nm_csv.read_text().splitlines()) == 1 + 3 * 9 * 9
    assert len(downstream_csv.read_text().splitlines()) == 1 + 3 * 3 * 9 * 4


def test_checkpoint_layouts_are_explicit(tmp_path):
    # Keep this lightweight: the path contract is what connects the evaluator to
    # the two archived training inventories on Tucker.
    from scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy import (
        checkpoint_path,
        evaluation_prefix,
    )

    arch = SimpleNamespace(
        checkpoint_layout="architecture-matrix",
        state_root=str(tmp_path),
        run_stamp="arch",
        checkpoint_step=100,
    )
    final = SimpleNamespace(
        checkpoint_layout="final-core",
        state_root=str(tmp_path),
        run_stamp="final",
        checkpoint_step=2500,
    )
    assert str(checkpoint_path(arch, 0, "ordA_r2")).endswith(
        "prodigy/archmatrix_prodigy_ordA_r2_s0_arch/checkpoint/state_dict_100.ckpt"
    )
    assert str(checkpoint_path(final, 2, "ordA_r2")).endswith(
        "finalcore_ordA_r2_s2_final/checkpoint/state_dict_2500.ckpt"
    )
    assert evaluation_prefix("ordA_r2", "covid_political", 2, 2500) == (
        "archmatrix_prodigy_eval_ordA_r2_s2_step2500_covid_political"
    )
    assert evaluation_prefix("ordA_r2", "covid_political", 0, 100) != (
        evaluation_prefix("ordA_r2", "covid_political", 2, 2500)
    )

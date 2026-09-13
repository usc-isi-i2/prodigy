import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze.py"
SPEC = importlib.util.spec_from_file_location("paper_mechanism_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def synthetic() -> pd.DataFrame:
    rows = []
    task_specs = (
        ("nm", analysis.NM_TARGETS, tuple(analysis.ARMS)),
        (
            "classification",
            analysis.CLS_TARGETS,
            tuple(name for name, (_, dim) in analysis.ARMS.items() if dim == 256),
        ),
    )
    for task, targets, arms in task_specs:
        for arm in arms:
            probability, emb_dim = analysis.ARMS[arm]
            for step in analysis.STEPS:
                for seed in analysis.SEEDS:
                    model_id = f"paper_mech_{arm}_step{step}_s{seed}"
                    for target in targets:
                        rows.append(
                            {
                                "task": task,
                                "arm": arm,
                                "cross_graph_prob": probability,
                                "emb_dim": emb_dim,
                                "checkpoint_step": step,
                                "training_seed": seed,
                                "dataset": target,
                                "roc_auc": 0.5 + step / 1_000_000 - probability * 0.01
                                + (emb_dim - 256) / 256 * 0.001,
                                "model_id": model_id,
                                "checkpoint": f"/{model_id}.ckpt",
                                "checkpoint_sha256": f"hash-{model_id}",
                                "training_revision": "revision",
                                "sources": analysis.EXPECTED_SOURCES,
                                "fingerprint": f"nm-{target}",
                                "episode_fingerprint": f"cls-{target}",
                            }
                        )
    return pd.DataFrame(rows)


def test_synthetic_campaign_passes_preregistered_gates():
    frame = synthetic()
    nm = frame[frame.task.eq("nm")].copy()
    cls = frame[frame.task.eq("classification")].copy()
    analysis.validate_grid(nm, "NM", analysis.NM_TARGETS, tuple(analysis.ARMS))
    analysis.validate_grid(
        cls,
        "classification",
        analysis.CLS_TARGETS,
        tuple(name for name, (_, dim) in analysis.ARMS.items() if dim == 256),
    )
    joined = analysis.provenance_join(nm, cls)
    per_seed, summary = analysis.macro_tables(frame)
    verdict = analysis.decisions(frame, per_seed)
    assert len(joined) == 75
    assert len(per_seed) == 165
    assert len(summary) == 55
    assert verdict["ratio"]["universal_winner"]
    assert verdict["data_scale"]["nm"]["positive"]
    assert verdict["capacity_nm"]["positive"]


def test_cross_task_checkpoint_drift_is_rejected():
    frame = synthetic()
    nm = frame[frame.task.eq("nm")].copy()
    cls = frame[frame.task.eq("classification")].copy()
    first = cls.index[0]
    cls.loc[first, "checkpoint_sha256"] = "different"
    try:
        analysis.provenance_join(nm, cls)
    except ValueError as error:
        assert "checkpoint_sha256" in str(error)
    else:
        raise AssertionError("expected cross-task provenance failure")

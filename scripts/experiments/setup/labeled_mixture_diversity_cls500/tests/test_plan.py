from math import comb
from pathlib import Path

from scripts.experiments.setup.labeled_mixture_diversity_cls500.make_plan import (
    TARGETS, evaluation_rows, render_eval, render_train, rows, validate,
)


def test_plan():
    plan = rows()
    validate(plan)
    assert len(plan) == sum(comb(5, k) for k in range(1, 5)) == 30
    assert len(evaluation_rows(plan)) == 75
    assert all(row["target"] not in row["donors"] for row in evaluation_rows(plan))


def test_manifests():
    here = Path(__file__).resolve().parents[1]
    assert (here / "manifest.tsv").read_text() == render_train()
    assert (here / "evaluation_manifest.tsv").read_text() == render_eval()


def test_training_uses_isolated_worker_processes():
    here = Path(__file__).resolve().parents[1]
    config = (here / "train.yaml").read_text()
    launcher = (here / "run_train_tucker.sh").read_text()
    assert "workers: 2\n" in config
    assert '--model-prefix "${prefix}"' in launcher
    assert "timeout --signal=TERM" in launcher

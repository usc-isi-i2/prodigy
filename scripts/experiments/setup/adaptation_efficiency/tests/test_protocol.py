import numpy as np

from scripts.experiments.setup.adaptation_efficiency.protocol import (
    LABEL_BUDGETS,
    COMMON_DIM,
    UPDATE_STEPS,
    fingerprint_model,
    new_head,
    run_curve,
    sampled_labels,
    stratified_node_splits,
)


def synthetic():
    labels = np.repeat(np.arange(2), 1000)
    features = np.column_stack((labels, 1 - labels, np.linspace(-1, 1, labels.size))).astype(np.float32)
    return features, labels, stratified_node_splits(labels, seed=0)


def test_label_samples_are_balanced_and_nested():
    _, labels, splits = synthetic()
    samples = {
        budget: sampled_labels(labels, splits["train"], budget=budget, seed=2)
        for budget in (1, 10, 100)
    }
    for budget, rows in samples.items():
        assert np.bincount(labels[rows]).tolist() == [budget, budget]
    assert set(samples[1]) <= set(samples[10]) <= set(samples[100])


def test_zero_labels_has_no_optimizer_updates():
    features, labels, splits = synthetic()
    rows = run_curve(
        features, labels, splits, model_id="encoder", target="toy", label_seed=0, budget=0
    )
    assert len(rows) == 2
    assert {row["head_updates"] for row in rows} == {0}
    assert {row["split"] for row in rows} == {"val", "test"}


def test_complete_valid_grid_and_reproducible_initialization():
    features, labels, splits = synthetic()
    rows = []
    for budget in LABEL_BUDGETS:
        rows.extend(
            run_curve(
                features, labels, splits, model_id="encoder", target="toy", label_seed=1, budget=budget
            )
        )
    cells = {(row["label_budget_per_class"], row["head_updates"], row["split"]) for row in rows}
    expected = {(0, 0, split) for split in ("val", "test")}
    expected |= {
        (budget, update, split)
        for budget in (1, 10, 100)
        for update in UPDATE_STEPS
        for split in ("val", "test")
    }
    assert cells == expected
    left = new_head("linear", COMMON_DIM, 2, 1)
    right = new_head("linear", COMMON_DIM, 2, 1)
    for a, b in zip(left.parameters(), right.parameters()):
        assert np.array_equal(a.detach().numpy(), b.detach().numpy())
    assert fingerprint_model(left) == fingerprint_model(right)

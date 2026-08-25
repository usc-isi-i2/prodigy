from math import comb
from pathlib import Path

from scripts.experiments.setup.nm_mixture_diversity_heldout_cls_2k.make_plan import (
    SOURCES,
    TARGET_SUBSETS,
    evaluation_rows,
    render_evaluation_manifest,
    render_manifest,
    rows,
    validate,
)


def test_exhaustive_balanced_heldout_plan():
    plan = rows()
    validate(plan)
    assert len(plan) == sum(comb(8, k) for k in range(1, 5))
    assert len(evaluation_rows(plan)) == len(TARGET_SUBSETS) * sum(
        comb(7, k) for k in range(1, 5)
    )
    for row in evaluation_rows(plan):
        assert row["target"] not in row["donors"]
        assert len(row["donors"]) == row["mixture_size"]


def test_every_donor_is_balanced_within_target_and_size():
    plan = evaluation_rows()
    for target in TARGET_SUBSETS:
        eligible = [source[1] for source in SOURCES if source[1] != target]
        for size in range(1, 5):
            group = [
                row for row in plan
                if row["target"] == target and row["mixture_size"] == size
            ]
            counts = [sum(donor in row["donors"] for row in group) for donor in eligible]
            assert len(set(counts)) == 1


def test_checked_in_manifest_is_current():
    here = Path(__file__).resolve().parents[1]
    assert (here / "manifest.tsv").read_text(encoding="utf-8") == render_manifest()
    assert (here / "evaluation_manifest.tsv").read_text(
        encoding="utf-8"
    ) == render_evaluation_manifest()

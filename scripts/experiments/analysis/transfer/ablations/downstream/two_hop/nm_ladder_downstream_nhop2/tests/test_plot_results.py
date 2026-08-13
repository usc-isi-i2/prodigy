from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plot_results import entry_summary, load_results  # noqa: E402


def test_completed_result_counts_and_headline_entry_summaries():
    entry, long, paired = load_results(ROOT / "data")

    assert len(entry) == 40
    assert long["logical_id"].nunique() == 40
    assert len(paired) == 216

    by_task = entry.groupby("task")["delta"].agg(
        n="size", positive=lambda values: int((values > 0).sum()), mean="mean"
    )
    assert tuple(by_task.loc["classification", ["n", "positive"]]) == (19, 9)
    assert tuple(by_task.loc["static_lp", ["n", "positive"]]) == (21, 19)

    summary = entry_summary(entry).set_index(["task", "variant", "order"])
    assert round(summary.loc[("static_lp", "fixed10k", "C"), "mean"], 6) == 0.071228
    assert summary.loc[("classification", "split", "A"), "positive"] == 1

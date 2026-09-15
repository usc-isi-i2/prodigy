#!/usr/bin/env python3
"""Independent arithmetic and decision audit for the H1 prerequisite."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
result = json.loads((HERE / "data/results.json").read_text())
assert result["complete"] and not result["test_2019_scored"]
assert len(result["rows"]) == 6
assert {(r["year"], r["seed"]) for r in result["rows"]} == {
    (year, seed) for year in (2017, 2018) for seed in (0, 1, 2)
}
for row in result["rows"]:
    np.testing.assert_allclose(row["gain_over_joint"], row["mixture_hits_at_50"] - row["joint_hits_at_50"])
    np.testing.assert_allclose(
        row["gain_over_better_single"],
        row["mixture_hits_at_50"] - max(row["joint_hits_at_50"], row["aa_hits_at_50"]),
    )
for year in (2017, 2018):
    rows = [r for r in result["rows"] if r["year"] == year]
    summary = result["summary"][str(year)]
    np.testing.assert_allclose(summary["mean_mixture"], np.mean([r["mixture_hits_at_50"] for r in rows]))
    np.testing.assert_allclose(summary["mean_joint"], np.mean([r["joint_hits_at_50"] for r in rows]))
    np.testing.assert_allclose(summary["mean_gain_over_joint"], np.mean([r["gain_over_joint"] for r in rows]))
expected = {
    "all_seed_gains_over_joint_positive_both_years": False,
    "mean_gain_over_joint_each_year": False,
    "forward_repeat_recall_loss_vs_aa_max": True,
    "forward_gain_over_better_single": False,
}
assert result["conditions"] == expected
assert not result["advance"] and result["decision"] == "stop H1 before training"
receipt = {
    "complete": True,
    "rows": 6,
    "years": [2017, 2018],
    "conditions_recomputed": expected,
    "decision_recomputed": "stop H1 before training",
    "test_2019_scored": False,
}
(HERE / "data/independent_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))


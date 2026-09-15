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

selector = json.loads((HERE / "data/selector_results.json").read_text())
assert selector["complete"] and not selector["test_2019_scored"]
assert len(selector["rows"]) == 3
assert [row["seed"] for row in selector["rows"]] == [0, 1, 2]
assert all(row["selected_rule"] == "joint_all_novel" for row in selector["rows"])
for row in selector["rows"]:
    candidates = row["all_selection_candidates"]
    chosen = max(candidates, key=lambda item: (item["hits_at_50"], -item["order"]))
    assert chosen["rule"] == row["selected_rule"]
    np.testing.assert_allclose(
        row["forward"]["gain_over_aa"],
        row["forward"]["hits_at_50"] - 0.668397576725917,
    )
selector_expected = {
    "forward_gain_over_aa_every_seed": False,
    "forward_repeat_recall_loss_vs_aa_max": True,
    "forward_novel_net_hits_positive_every_seed": False,
}
assert selector["conditions"] == selector_expected
assert not selector["advance"] and selector["decision"] == "stop selector; do not train gate"
selector_receipt = {
    "complete": True,
    "rows": 3,
    "selected_rules_recomputed": ["joint_all_novel"] * 3,
    "conditions_recomputed": selector_expected,
    "decision_recomputed": "stop selector; do not train gate",
    "test_2019_scored": False,
}
(HERE / "data/selector_independent_audit.json").write_text(json.dumps(selector_receipt, indent=2) + "\n")
print(json.dumps(selector_receipt, indent=2))

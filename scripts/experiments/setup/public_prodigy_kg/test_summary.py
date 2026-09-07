"""Metric orientation, example accounting and complete-arm contract."""
import copy

import torch

from .summarize_mechanism import ARMS, VIEWS, score_episode, summarize_rows


def test_summary():
    wrong = torch.tensor([[0., 2.], [2., 0.]])
    right = wrong.flip(1)
    result = {"y_true": torch.eye(2),
              "parity": {"max_abs_error": 0, "atol": 0, "rtol": 0},
              "arms": {arm: {**{view: wrong.clone() for view in VIEWS},
                             "query_max_abs_change": 0., "reference_max_abs_change": 0.}
                       for arm in ARMS}}
    result["arms"]["values"]["logits"] = right
    result["arms"]["values"]["reference_only_logits"] = right
    rows, examples = score_episode(0, result)
    summary = summarize_rows(rows, draws=20)
    assert summary["metrics"]["accuracy"]["primary_value_minus_key"]["mean_delta"] == 1
    assert summary["metrics"]["macro_f1"]["primary_value_minus_key"]["mean_delta"] == 1
    assert summary["metrics"]["nll"]["primary_value_minus_key"]["mean_delta"] < 0
    assert len(examples) == 2 * len(ARMS) * len(VIEWS)
    corrected = next(row for row in rows if row["arm"] == "values" and row["view"] == "logits")
    assert corrected["corrected_queries"] == 2 and corrected["corrupted_queries"] == 0
    missing = copy.deepcopy(result)
    del missing["arms"]["keys"]
    try:
        score_episode(0, missing)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing arm accepted")
    try:
        summarize_rows(rows + rows, draws=20)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate episodes accepted")
    print("Mechanism summary tests passed")


if __name__ == "__main__":
    test_summary()

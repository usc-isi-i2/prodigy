import numpy as np
import torch

from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_health_predictiveness import (
    model_rows,
)


def test_model_rows_reports_label_free_health_and_mapping_aware_auc():
    mapping = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
    local_y = torch.tensor([0, 0, 1, 1])
    full = torch.tensor([[3., 0.], [3., 0.], [0., 3.], [0., 3.]])
    u1 = torch.tensor([[3., 0.], [3., 0.], [3., 0.], [0., 3.]])
    record = {"labels": {"local_y": local_y, "mapping": mapping, "use_global": True},
              "models": {"m": {"logits": {"full_model": full, "U1_pre_meta/ridge": u1,
                                              "raw_joint/ridge": u1}}}}
    row = model_rows(record, "target", "fresh")[0]
    assert row["u1_agreement_rate"] == .75
    assert row["accuracy"] == 1.0
    assert row["auc"] == 1.0

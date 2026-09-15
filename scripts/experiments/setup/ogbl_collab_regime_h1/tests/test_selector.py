import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("selector", ROOT / "selector.py")
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


def test_contract_is_bounded_and_forward_only():
    cfg = SELECTOR.contract()
    assert len(cfg["fixed_rule_order"]) == 14
    assert cfg["selection_year"] == 2017 and cfg["forward_year"] == 2018
    assert cfg["test_2019_access"] is False


def test_pooled_threshold_uses_only_novel_candidates():
    panel = {
        "pfeatures": np.zeros((2, 13)),
        "nfeatures": np.zeros((2, 13)),
    }
    panel["pfeatures"][:, 7] = [1, 100]
    panel["nfeatures"][:, 7] = [3, 200]
    panel["pfeatures"][:, 8] = [0, 1]
    panel["nfeatures"][:, 8] = [0, 1]
    assert SELECTOR.pooled_threshold(panel, 7, .5) == 2


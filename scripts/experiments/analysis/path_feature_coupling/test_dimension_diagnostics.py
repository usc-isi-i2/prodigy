from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "dimension_diagnostics", MODULE_DIR / "analyze_dimension_diagnostics.py"
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_column_pearson_recovers_signed_dimensions() -> None:
    target = np.arange(20, dtype=np.float64)
    values = np.stack((target, -target, np.ones_like(target)), axis=1)
    corr = MOD.column_pearson(values, target)
    assert np.isclose(corr[0], 1.0)
    assert np.isclose(corr[1], -1.0)
    assert np.isnan(corr[2])


def test_component_certificate_rules_out_distance_1000() -> None:
    from scipy.sparse import csr_matrix

    # A 1,200-node star is large enough that size alone cannot rule out 1,000,
    # but root eccentricity 1 gives the valid diameter upper bound 2.
    leaves = np.arange(1, 1_200)
    rows = np.concatenate((np.zeros_like(leaves), leaves))
    cols = np.concatenate((leaves, np.zeros_like(leaves)))
    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(1_200, 1_200))
    result = MOD.component_distance_certificate(adjacency)
    assert result["global_diameter_upper_bound"] == 2
    assert result["exact_distance_1000_certified_absent"] is True

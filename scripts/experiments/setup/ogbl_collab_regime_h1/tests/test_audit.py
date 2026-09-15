import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "audit.py"
SPEC = importlib.util.spec_from_file_location("regime_h1_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_cdf_is_monotonic_and_frozen():
    ref = AUDIT.fit_cdf(np.array([3.0, 1.0, 2.0]))
    got = AUDIT.apply_cdf(ref, np.array([0.0, 1.0, 1.5, 3.0, 4.0]))
    np.testing.assert_allclose(got, [0.0, 1 / 3, 1 / 3, 1.0, 1.0])


def test_hits_uses_strict_shared_cutoff():
    positives = np.array([1.0, 2.0, 3.0])
    negatives = np.arange(100, dtype=float)
    value, mask, cutoff = AUDIT.hits(positives, negatives)
    assert cutoff == 50.0
    assert value == 0.0
    assert not mask.any()


def test_protocol_forbids_test_access():
    cfg = AUDIT.protocol()
    assert cfg["test_2019_access"] is False
    assert cfg["gate"].startswith("AA-DC if pair appeared")


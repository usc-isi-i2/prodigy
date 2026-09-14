import pytest

from mixture_scaling.analyze import STEPS, log_step_aulc, validate_aggregate


def test_log_step_aulc_preserves_constant_score():
    assert log_step_aulc({step: 0.73 for step in STEPS}) == pytest.approx(0.73)


def test_log_step_aulc_requires_all_declared_checkpoints():
    with pytest.raises(ValueError, match="expected steps"):
        log_step_aulc({100: 0.5, 300: 0.6, 900: 0.7})


def test_validate_aggregate_rejects_missing_checkpoints():
    manifest = [{"run_id": "specialist_g_s0", "target": "g", "sources": "g", "seed": "0"}]
    with pytest.raises(ValueError, match="missing 4 results"):
        validate_aggregate([], manifest, "test")

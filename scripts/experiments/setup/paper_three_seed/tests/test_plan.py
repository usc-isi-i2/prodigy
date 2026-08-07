from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_plan import build_plan  # noqa: E402


def test_registered_plan_shape_and_groups():
    jobs = build_plan()
    assert len(jobs) == 57
    assert len({(item.family, item.arm) for item in jobs}) == 57
    assert sum(item.eval_group == "sage_1hop" for item in jobs) == 26
    assert sum(item.eval_group == "gat_1hop" for item in jobs) == 8
    assert sum(item.eval_group == "sage_2hop" for item in jobs) == 23


def test_fixed_exposure_shared_endpoint_is_not_duplicated():
    jobs = build_plan()
    fixed = [item.arm for item in jobs if item.family == "fixed_exposure_2hop"]
    assert len(fixed) == 15
    assert "ordA_r8" in fixed
    assert "ordC_r8" not in fixed


def test_fixed_exposure_uses_registered_terminal_steps():
    jobs = build_plan()
    steps = {
        item.arm: item.target_step
        for item in jobs
        if item.family == "fixed_exposure_2hop"
    }
    assert steps["ordA_r1"] == 10_000
    assert steps["ordA_r8"] == 80_000
    assert steps["ordC_r7"] == 70_000

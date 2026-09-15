import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ogbl_collab_compact_joint"))
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("multiyear", ROOT / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_contract_is_bounded_and_test_closed():
    cfg = RUN.contract()
    assert cfg["training_years"] == [2015, 2016, 2017]
    assert cfg["test_2019_access"] is False
    assert "all3" in cfg["advance"] and "mean gain>=.005" in cfg["advance"]


def test_schedule_is_balanced():
    visits = {year: 0 for year in RUN.YEARS}
    for step in range(1, RUN.STEPS + 1):
        visits[RUN.YEARS[(step - 1) % 3]] += 1
    assert visits == {2015: 667, 2016: 667, 2017: 666}


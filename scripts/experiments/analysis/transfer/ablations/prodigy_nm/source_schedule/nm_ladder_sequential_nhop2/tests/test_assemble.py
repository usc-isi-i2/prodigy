from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "assemble_results.py"
SPEC = importlib.util.spec_from_file_location("seq_assemble", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_eval_row_prefers_newest_complete_run(tmp_path: Path) -> None:
    prefix = "nm_ladder_seq_h2m_r2"
    old = tmp_path / f"eval_{prefix}_to_covid19_twitter_nm_3shot_30way_old" / "data"
    new = tmp_path / f"eval_{prefix}_to_covid19_twitter_nm_3shot_30way_new" / "data"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "metrics_test_step0.json").write_text(json.dumps({"test_roc_auc": 0.6}))
    (new / "metrics_test_step0.json").write_text(json.dumps({"test_roc_auc": 0.7}))
    old.parent.touch()
    new.parent.touch()
    values, provenance = MODULE.eval_row(tmp_path, prefix)
    assert values["covid19_twitter"] == 0.7
    assert provenance["covid19_twitter"].endswith("_new")


def test_pairing_assigns_schedule_roles() -> None:
    rows = [
        {"rung": 3, "test_graph": "covid19_twitter", "auc": 0.8,
         "entry_rung": 2, "rel_to_entry": 1, "in_training": 1, "is_newcomer": 0},
        {"rung": 3, "test_graph": "midterm", "auc": 0.9,
         "entry_rung": 3, "rel_to_entry": 0, "in_training": 1, "is_newcomer": 1},
        {"rung": 3, "test_graph": "twibot20", "auc": 0.6,
         "entry_rung": 7, "rel_to_entry": -4, "in_training": 0, "is_newcomer": 0},
    ]
    control = {
        (3, "covid19_twitter"): 0.75,
        (3, "midterm"): 0.85,
        (3, "twibot20"): 0.65,
    }
    paired = MODULE.pair_with_control(rows, control)
    assert [row["role"] for row in paired] == ["incumbent", "newcomer", "heldout"]

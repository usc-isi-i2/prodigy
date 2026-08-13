import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "assemble_results.py"
SPEC = importlib.util.spec_from_file_location("assemble_tts", MODULE)
assemble = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assemble)


def test_latest_complete_retry_wins(tmp_path):
    old = tmp_path / "eval_nm_ladder_tts_h2m_r1_to_midterm_nm_3shot_30way_old"
    (old / "data").mkdir(parents=True)
    (old / "data" / "metrics_test_step0.json").write_text(
        json.dumps({"test_roc_auc": 0.7})
    )
    new = tmp_path / "eval_nm_ladder_tts_h2m_r1_to_midterm_nm_3shot_30way_new"
    (new / "data").mkdir(parents=True)
    (new / "data" / "metrics_test_step0.json").write_text(
        json.dumps({"test_roc_auc": 0.8})
    )
    values, provenance = assemble.eval_row(tmp_path, "nm_ladder_tts_h2m_r1")
    assert values["midterm"] == 0.8
    assert provenance["midterm"] == new.name


def test_partial_assembly_has_registered_8x8_shape(tmp_path):
    wide, long_rows, missing = assemble.assemble(tmp_path)
    assert len(wide) == 8
    assert len(long_rows) == 64
    assert len(missing) == 8
    assert {row["context_view"] for row in long_rows} == {"static_background"}
    assert {row["positive_view"] for row in long_rows} == {"static_holdout"}

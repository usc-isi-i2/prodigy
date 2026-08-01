from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_configs.py"
SPEC = importlib.util.spec_from_file_location("sat_h2_validate", MODULE_PATH)
validate_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate_module)


def test_registered_configs_are_exact() -> None:
    assert validate_module.validate(MODULE_PATH.parent) == []


def test_validator_rejects_radius_drift(tmp_path: Path) -> None:
    for arm in validate_module.ARM_FIELDS:
        source = MODULE_PATH.parent / f"train_{arm}.yaml"
        text = source.read_text(encoding="utf-8")
        if arm == "ukr":
            text = text.replace("n_hop: 2", "n_hop: 1")
        (tmp_path / source.name).write_text(text, encoding="utf-8")
    errors = validate_module.validate(tmp_path)
    assert any("n_hop=1" in error for error in errors)

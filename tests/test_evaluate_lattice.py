from pathlib import Path

from mixture_scaling.evaluate_lattice import CLS_TARGETS, LP_TARGETS, load_pair_module


def test_target_contracts():
    assert len(CLS_TARGETS) == 5
    assert len(LP_TARGETS) == 6
    assert "facebook_page_reference" in CLS_TARGETS
    assert "facebook_page_reference" in LP_TARGETS


def test_actual_repaired_pair_evaluator_imports_when_repo_is_available():
    root = Path(__file__).resolve().parents[2] / "prodigy"
    if root.is_dir():
        module = load_pair_module(root)
        assert module.self_test(verbose=False)

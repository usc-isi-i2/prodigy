from mixture_scaling.evaluate_lattice import CLS_TARGETS, LP_TARGETS
from mixture_scaling.lattice import lattice_rows


def test_expected_completion_cardinalities():
    assert len(lattice_rows()) == 54
    assert len(lattice_rows()) * len(CLS_TARGETS) == 270
    assert len(lattice_rows()) * len(LP_TARGETS) == 324

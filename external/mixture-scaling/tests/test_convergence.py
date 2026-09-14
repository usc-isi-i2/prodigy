import pytest
from mixture_scaling.convergence import SourcePlateau


def test_improving_source_prevents_stop_even_if_other_source_is_flat():
    stopper = SourcePlateau(["a", "b"], patience=2, min_delta=.001)
    assert not stopper.update({"a": .7, "b": .7}, True)
    assert not stopper.update({"a": .7, "b": .6}, True)
    assert not stopper.update({"a": .7, "b": .5}, True)
    assert not stopper.update({"a": .7, "b": .5}, True)
    assert stopper.update({"a": .7, "b": .5}, True)


def test_minimum_exposure_and_cumulative_improvements():
    stopper = SourcePlateau(["a"], patience=2, min_delta=.01)
    for _ in range(5):
        assert not stopper.update({"a": .7}, False)
    assert not stopper.update({"a": .695}, True)
    assert not stopper.update({"a": .68}, True)
    assert not stopper.update({"a": .68}, True)
    assert stopper.update({"a": .68}, True)


def test_nonfinite_validation_fails_instead_of_looking_converged():
    with pytest.raises(FloatingPointError):
        SourcePlateau(["a"]).update({"a": float("nan")}, True)

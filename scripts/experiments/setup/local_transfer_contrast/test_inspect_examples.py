from scripts.experiments.setup.local_transfer_contrast.inspect_examples import locate_occurrence


def test_locate_occurrence_across_variable_batches():
    assert locate_occurrence([3, 2, 4], 0) == (0, 0)
    assert locate_occurrence([3, 2, 4], 3) == (1, 0)
    assert locate_occurrence([3, 2, 4], 8) == (2, 3)

from mixture_scaling.lattice import SOURCE_ORDER, lattice_rows, selected_rows, update_selection


def test_lattice_is_9_plus_36_plus_9():
    rows = lattice_rows()
    assert len(SOURCE_ORDER) == 9
    assert len(rows) == 54
    assert len({name for name, _ in rows}) == 54
    assert sorted(map(lambda row: len(row[1]), rows)).count(1) == 9
    assert sorted(map(lambda row: len(row[1]), rows)).count(2) == 36
    assert sorted(map(lambda row: len(row[1]), rows)).count(8) == 9


def test_gate_has_three_shape_representatives():
    assert sorted(len(sources) for _, sources in selected_rows("gate")) == [1, 2, 8]


def test_absolute_best_is_saved_without_patience_reset():
    result = update_selection(0.999, 250, 1.0, 0, 1.0, 0, 0.0025)
    best_loss, best_step, reference, patience, save_best = result
    assert (best_loss, best_step, save_best) == (0.999, 250, True)
    assert reference == 1.0
    assert patience == 1

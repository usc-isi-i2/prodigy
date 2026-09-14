from mixture_scaling.train_strict import update_selection


def test_minor_absolute_improvement_saves_without_resetting_patience():
    state = update_selection(0.6400, 4000, 0.6402, 3750, 0.6402, 1, 0.002)
    best_loss, best_step, reference, patience, save_best = state
    assert save_best
    assert best_loss == 0.6400
    assert best_step == 4000
    assert reference == 0.6402
    assert patience == 2


def test_meaningful_improvement_saves_and_resets_patience():
    state = update_selection(0.6380, 4250, 0.6402, 3750, 0.6402, 3, 0.002)
    assert state == (0.6380, 4250, 0.6380, 0, True)

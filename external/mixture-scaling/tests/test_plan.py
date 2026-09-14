from mixture_scaling.plan import ladder_rows, matrix_rows, rows


def config():
    return {"graphs": {name: {} for name in ("a", "b", "c", "d")}}


def test_primary_has_specialist_and_loo_per_target():
    result = rows(config(), [0])
    assert len(result) == 8
    assert all(row["target"] not in row["sources"].split(",") for row in result if row["kind"] == "leave_one_out")


def test_ladder_is_target_confined_and_has_all_sizes():
    training, evaluations = ladder_rows(config(), [0])
    assert len(evaluations) == 4
    assert {row["kind"] for row in evaluations} == {"ladder_k2"}
    assert all(row["target"] not in row["sources"].split(",") for row in evaluations)
    assert len(training) <= len(evaluations)


def test_matrix_maps_every_primary_model_to_every_target():
    result = matrix_rows(config(), [0])
    assert len(result) == 32

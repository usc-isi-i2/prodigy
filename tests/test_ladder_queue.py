from mixture_scaling.ladder_queue import claimed_rows


def test_queue_claims_largest_first_without_double_claiming(tmp_path):
    rows=[("r1", ("a",)), ("r2", ("a","b")), ("r3", ("a","b","c"))]
    first=claimed_rows(tmp_path,rows)
    assert next(first)[0]=="r3"
    second=claimed_rows(tmp_path,rows)
    assert next(second)[0]=="r2"
    first.close()
    third=claimed_rows(tmp_path,rows)
    assert next(third)[0]=="r3"
    second.close()
    third.close()


def test_queue_skips_completed_models(tmp_path):
    run=tmp_path/"lp"/"r1"
    run.mkdir(parents=True)
    (run/"summary.json").write_text('{}')
    assert list(claimed_rows(tmp_path,[("r1",("a",))]))==[]

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


def test_exited_unreaped_worker_does_not_block_handoff(tmp_path):
    from mixture_scaling.schedule_ladder import process_running
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "stat").write_text("123 (python worker) Z 1 0 0")
    assert not process_running(123, tmp_path)
    (proc / "stat").write_text("123 (python worker) R 1 0 0")
    assert process_running(123, tmp_path)
    assert not process_running(456, tmp_path)

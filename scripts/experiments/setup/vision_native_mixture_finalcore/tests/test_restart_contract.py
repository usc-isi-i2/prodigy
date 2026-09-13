from pathlib import Path


def test_replica_launcher_uses_explicit_run_name_and_archives_partial_results():
    script = (
        Path(__file__).resolve().parents[1] / "run_tucker.sh"
    ).read_text(encoding="utf-8")
    assert '--model-ids "$model_id" --run-name "$model_id"' in script
    assert '"${result}.incomplete_${failed_stamp}"' in script
    assert '"$(wc -l < "$result")" -ne 5' in script

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FINAL_CORE = HERE.parent / "final_core"
sys.path.insert(0, str(FINAL_CORE))

import evaluate_fixed_grid  # noqa: E402


def parse_with(monkeypatch, tmp_path: Path, extra: list[str]):
    argv = [
        "evaluate_fixed_grid.py",
        "--worker-index", "0",
        "--worker-count", "1",
        "--training-state-root", str(tmp_path / "training"),
        "--evaluation-state-root", str(tmp_path / "state"),
        "--evaluation-log-root", str(tmp_path / "log"),
        "--results-root", str(tmp_path / "results"),
        "--evaluation-run-stamp", "contract",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return evaluate_fixed_grid.parse_args()


def test_original_executor_defaults_are_preserved(monkeypatch, tmp_path: Path) -> None:
    args = parse_with(monkeypatch, tmp_path, [])
    assert args.eval_split == "test"
    assert args.checkpoint_step == 2500
    assert args.target_edge_view == "static_test"
    assert args.protocol == "fixed_test_512_static_test_on_static_train_v1"


def test_pilot_executor_resolves_validation_only(monkeypatch, tmp_path: Path) -> None:
    args = parse_with(
        monkeypatch,
        tmp_path,
        [
            "--eval-split", "val",
            "--checkpoint-step", "300",
            "--protocol", "pilot_transfer_selection_v1",
            "--config", str(HERE / "evaluation.yaml"),
            "--batch-size", "32",
        ],
    )
    params = evaluate_fixed_grid.resolved_params(
        args,
        seed=0,
        model_id="ss_ukr_rus",
        target="covid",
        checkpoint=Path("/state/state_dict_300.ckpt"),
    )
    assert args.target_edge_view == "static_validation"
    assert params["eval_only"] is True
    assert params["eval_only_split"] == "val"
    assert params["target_edge_view"] == "static_validation"
    assert params["val_len_cap"] == 16
    assert params["neighbor_sampling_source_subset"] == "covid"


def test_validation_rejects_test_fingerprint_ledger(monkeypatch, tmp_path: Path) -> None:
    try:
        parse_with(
            monkeypatch,
            tmp_path,
            [
                "--eval-split", "val",
                "--reference-fingerprints", str(tmp_path / "test.tsv"),
            ],
        )
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("validation must reject the terminal test ledger")

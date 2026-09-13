from pathlib import Path


SETUP = Path(__file__).resolve().parents[2]
SCRIPTS = (
    SETUP / "paper_three_seed/recover_flagship_after_remainder_tucker.sh",
    SETUP / "paper_three_seed/run_fast_core_eval_tucker.sh",
    SETUP / "paper_mechanism_sweeps/wait_and_run_tucker.sh",
    SETUP / "paper_all_pairs/wait_and_run_tucker.sh",
)


def test_overnight_handoffs_are_restricted_to_gpus_2_and_3() -> None:
    forbidden = (
        "for gpu in 0 1 2 3",
        'GPUS="0 2 3"',
        'GPUS="1"',
        'echo "0 2 3"',
        'echo "1 2 3"',
        '[[ "$gpu" =~ ^[0-3]$ ]]',
    )
    for path in SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert '[[ "$gpu" =~ ^[23]$ ]]' in text
        for token in forbidden:
            assert token not in text, f"{path.name} retains forbidden GPU path: {token}"


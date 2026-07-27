"""Shared arm/step/key definitions for the pretrain-saturation experiment.

Two setup folders feed one analysis folder, and they MUST agree on the model keys or
the 18 rows will not join:

  setup/pretrain_saturation_existing/  -> steps 1000, 2000, 10000, 40000 (no training)
  setup/pretrain_saturation_dense/     -> steps 100, 500 (three short retrains)
  analysis/pretrain_saturation/        -> the joined curve

This module is the single definition of both. ``pretrain_saturation_dense`` imports it
by path (same pattern as ``nm_ladder_downstream/make_model_list.py`` importing
``nm_ladder_order_robustness-jul_23/make_configs.py``); it lives here because this is the
folder that owns the arm -> historical-run mapping.

Why these three arms: they are the only NM runs whose full checkpoint trajectory survived
on Tucker (verified read-only 2026-07-27). ``all8`` is the top rung of the interpolation
ladder; ``ukr``/``covid`` are its single-source rungs, i.e. the narrow-corpus contrast.
"""
from __future__ import annotations

from pathlib import Path

# TWO state dirs, and conflating them is a real failure mode (it broke the first
# check_splice.py run). `state/` is gitignored and lives PER WORKTREE -- it does not
# follow a branch -- so:
#
#   historical checkpoints -> the main checkout, where those runs were trained
#   dense checkpoints      -> whichever worktree ran run_all_train_tucker.sh
#
# The historical location is absolute because it is a fact about where past jobs ran.
# The dense location is derived from this file's own path, so it is automatically correct
# whether the experiment runs from the main checkout or from a worktree.
DEFAULT_HISTORICAL_STATE_DIR = "/dataMeR1/phil/gfm/prodigy/state"

# setup/<name>/arms.py -> up 4 = repo root
REPO_ROOT = Path(__file__).resolve().parents[4]


def default_dense_state_dir() -> Path:
    """`state/` of the checkout this file belongs to."""
    return REPO_ROOT / "state"

# Steps served by each folder. Their union is the 18-checkpoint curve.
EXISTING_STEPS = (1000, 2000, 10000, 40000)
DENSE_STEPS = (100, 500)
ALL_STEPS = tuple(sorted(set(EXISTING_STEPS) | set(DENSE_STEPS)))

# Splice-validation probes written by the dense runs only. These are NOT evaluated;
# check_splice.py compares them tensor-for-tensor against the historical checkpoints.
# Pre-2026-07-26 runs named a checkpoint by the pre-increment loop variable, so a
# historical `state_dict_N` holds N+1 completed steps -- hence 1001 vs 1000.
SPLICE_PROBES = {1001: 1000, 2001: 2000}   # dense step -> historical step it should equal


class Arm:
    """One pretraining corpus: its historical run, and its dense-retrain counterpart."""

    def __init__(self, name, run_dir, train_config, dense_config, dense_prefix,
                 eval_step_in_history, note):
        self.name = name
        # Pinned exact run directory, not a prefix glob: `ukr_only_nm_*` also matches
        # `ukr_only_nm_aug_*`, and these are frozen historical runs anyway.
        self.run_dir = run_dir
        self.train_config = train_config
        self.dense_config = dense_config
        self.dense_prefix = dense_prefix
        # The historical run's eval_step. Determines which splice probes are comparable:
        # an in-loop val eval consumes global torch RNG (Collator -> linearize ->
        # torch.rand), so the training stream diverges after the first eval fires.
        self.eval_step_in_history = eval_step_in_history
        self.note = note

    def historical_ckpt(self, step, state_dir=DEFAULT_HISTORICAL_STATE_DIR):
        return Path(state_dir) / self.run_dir / "checkpoint" / f"state_dict_{step}.ckpt"

    def splice_probe_is_comparable(self, historical_step):
        """True when a no-val-eval dense run should reproduce this historical checkpoint.

        In the training loop the periodic save comes BEFORE the eval block in the same
        iteration, and a historical `state_dict_N` was written at loop index e=N. So it
        is unaffected by eval RNG iff no eval fired at any e < N. Evals fire at
        e = eval_step, 2*eval_step, ..., hence the condition is eval_step >= N.

        Worked example: ukr had eval_step=1000. `state_dict_1000` was written at e=1000
        just before that iteration's first-ever eval, so it IS clean; `state_dict_2000`
        was written after it, so it is not.
        """
        return historical_step <= self.eval_step_in_history

    def __repr__(self):
        return f"Arm({self.name})"


ARMS = (
    Arm(
        name="all8",
        run_dir="merged_ukr_rus_covid_midterm_all8_nm_wb_09_07_2026_15_10_30",
        train_config="scripts/experiments/setup/covid_ukr/merged_ukr_rus_covid_midterm_all8_nm.yaml",
        dense_config="train_all8_dense.yaml",
        dense_prefix="sat_dense_all8",
        eval_step_in_history=100_000,
        note="Top ladder rung: 8 merged sources, graph_id/balanced episode confinement. "
             "Trained 2026-07-09; survives 1000..43000.",
    ),
    Arm(
        name="ukr",
        run_dir="ukr_only_nm_14_06_2026_16_39_00",
        train_config="scripts/experiments/setup/ukr_only/ukr_only_nm.yaml",
        dense_config="train_ukr_dense.yaml",
        dense_prefix="sat_dense_ukr",
        eval_step_in_history=1_000,
        note="Ladder rung 1: ukr_rus_twitter alone. Trained 2026-06-14; survives 1000..119000.",
    ),
    Arm(
        name="covid",
        run_dir="covid_only_nm_14_06_2026_16_38_56",
        train_config="scripts/experiments/setup/covid_only/covid_only_nm.yaml",
        dense_config="train_covid_dense.yaml",
        dense_prefix="sat_dense_covid",
        eval_step_in_history=1_000,
        note="Second single-source rung: covid19_twitter alone. Trained 2026-06-14; "
             "survives 1000..119000.",
    ),
)

ARMS_BY_NAME = {arm.name: arm for arm in ARMS}


def model_key(arm_name: str, step: int) -> str:
    """Row key in the shared per-task eval CSVs.

    Zero-padded to six digits so a lexical sort of the CSV is also a numeric sort of the
    trajectory -- the analysis plots steps on the x axis and must not reorder 500 after
    40000.
    """
    return f"sat_{arm_name}_s{step:06d}"


def parse_model_key(key: str) -> tuple[str, int]:
    """Inverse of model_key; raises on anything that is not one of ours."""
    if not key.startswith("sat_"):
        raise ValueError(f"not a saturation model key: {key!r}")
    arm_name, _, step_part = key[len("sat_"):].rpartition("_s")
    if not step_part.isdigit() or arm_name not in ARMS_BY_NAME:
        raise ValueError(f"not a saturation model key: {key!r}")
    return arm_name, int(step_part)

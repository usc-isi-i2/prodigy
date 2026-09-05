#!/usr/bin/env python3
"""Run the final-core cached evaluator against the 36 pair checkpoints."""

from __future__ import annotations

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FINAL_CORE = HERE.parent / "final_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FINAL_CORE))

import pair_plan  # noqa: E402
import evaluate_fixed_grid as evaluator  # noqa: E402


evaluator.physical_jobs = pair_plan.physical_jobs
evaluator.checkpoint_path = pair_plan.checkpoint_path


if __name__ == "__main__":
    raise SystemExit(evaluator.main())

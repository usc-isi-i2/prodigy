#!/usr/bin/env python3
"""Prove that the dense retrains lie on the same trajectory as the historical runs.

The 18-checkpoint curve is spliced: steps 100 and 500 come from retrains done today,
steps 1000..40000 from runs trained on 2026-06-14 (ukr, covid) and 2026-07-09 (all8). If
the intervening code changed the optimization path, the joint between 500 and 1000 would
show a discontinuity that reads as -- or hides -- saturation. This script tests the
splice directly instead of arguing about it.

The test exploits the checkpoint-naming fix. The pre-2026-07-26 trainer named an in-loop
save by the pre-increment loop variable, so a historical ``state_dict_1000`` holds 1001
completed steps. Our dense runs therefore also write ``state_dict_1001``, and the two
files should be the same model.

    historical state_dict_1000   ==   dense state_dict_1001      (1001 steps each)
    historical state_dict_2000   ==   dense state_dict_2001      (2001 steps each)

The 2001 probe is only meaningful for an arm whose historical run did NOT run an in-loop
val eval before that point: an eval consumes global torch RNG (Collator -> linearize ->
torch.rand), so the two training streams legitimately diverge afterwards. ukr and covid
used eval_step=1000 and are skipped at 2001 by design; all8 used 100000 and is checked.
This is reported, not silently dropped.

Exact equality is not expected -- these ran on different days on possibly different GPUs,
and scatter/segment kernels are nondeterministic. The verdict compares the observed
difference against a reference distance: how far the historical run itself moved between
its own step 1000 and step 2000. Same trajectory => difference orders of magnitude
smaller than that.

Note the TWO state dirs. `state/` is gitignored and per-worktree, so the historical
checkpoints sit in the main checkout while the dense ones sit in whichever worktree ran
the retrains. The defaults handle this (dense = this checkout's `state/`), but they are
separate knobs on purpose.

Usage (on Tucker, after run_all_train_tucker.sh):
    python3 check_splice.py
    python3 check_splice.py --arm ukr
    HISTORICAL_STATE_DIR=... DENSE_STATE_DIR=... python3 check_splice.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXISTING = HERE.parent / "pretrain_saturation_existing"
# APPEND, never insert(0): both setup folders contain a `make_model_list.py`, and this
# script's own directory is already sys.path[0]. Inserting the sibling folder at the front
# shadows our module with theirs, and `resolve_dense_run_dir` then fails to import.
sys.path.append(str(EXISTING))

from arms import (  # noqa: E402
    ARMS_BY_NAME, DEFAULT_HISTORICAL_STATE_DIR, SPLICE_PROBES, default_dense_state_dir,
)
from make_model_list import resolve_dense_run_dir  # noqa: E402

# A difference this many times smaller than the run's own 1000-step displacement is
# float noise, not a different trajectory.
SAME_TRAJECTORY_RATIO = 100.0


def flat_tensors(blob, torch):
    """Flatten {module: state_dict} into {qualified_name: tensor}, floats only."""
    out = {}
    for module_name, state in sorted(blob.items()):
        if not hasattr(state, "items"):
            continue
        for param_name, tensor in sorted(state.items()):
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                out[f"{module_name}.{param_name}"] = tensor.detach().float().cpu()
    return out


def distance(a, b, torch):
    """(max abs diff, L2 norm of the difference) over the shared parameter set."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None, None, []
    only = sorted(set(a) ^ set(b))
    max_abs = 0.0
    sq = 0.0
    for key in keys:
        if a[key].shape != b[key].shape:
            only.append(f"{key} (shape {tuple(a[key].shape)} vs {tuple(b[key].shape)})")
            continue
        diff = (a[key] - b[key])
        max_abs = max(max_abs, float(diff.abs().max()))
        sq += float((diff ** 2).sum())
    return max_abs, sq ** 0.5, only


def main() -> int:
    ap = argparse.ArgumentParser()
    # The whole point of this script is to compare across TWO state dirs. They are almost
    # never the same directory: the historical runs live in the main checkout, the dense
    # retrains in whichever worktree ran them, because state/ is per-worktree.
    ap.add_argument("--historical-state-dir",
                    default=os.environ.get("HISTORICAL_STATE_DIR", DEFAULT_HISTORICAL_STATE_DIR),
                    help="Where the historical runs were trained (main checkout).")
    ap.add_argument("--dense-state-dir",
                    default=os.environ.get("DENSE_STATE_DIR", str(default_dense_state_dir())),
                    help="Where the dense retrains wrote (default: this checkout's state/).")
    ap.add_argument("--arm", action="append", choices=sorted(ARMS_BY_NAME),
                    help="Check only this arm (repeatable). Default: all three.")
    args = ap.parse_args()

    import torch

    hist_dir = Path(args.historical_state_dir)
    dense_dir_root = Path(args.dense_state_dir)
    print(f"historical state dir: {hist_dir}")
    print(f"dense state dir:      {dense_dir_root}")
    if hist_dir == dense_dir_root:
        print("[warn] both state dirs are identical -- is that really right? The dense "
              "runs normally live in a different worktree than the historical ones.")
    print()
    arms = [ARMS_BY_NAME[n] for n in (args.arm or sorted(ARMS_BY_NAME))]

    failures: list[str] = []
    skipped: list[str] = []

    for arm in arms:
        print("=" * 78)
        print(f"ARM {arm.name}   historical={arm.run_dir}")
        print("=" * 78)

        dense_dir = resolve_dense_run_dir(dense_dir_root, arm.dense_prefix)
        if dense_dir is None:
            failures.append(f"{arm.name}: no dense run dir {arm.dense_prefix}_*")
            print(f"  MISSING dense run dir {arm.dense_prefix}_* under {dense_dir_root}")
            continue
        print(f"  dense={dense_dir.name}")

        # Reference scale: how far this run moved over its own steps 1000 -> 2000.
        ref_norm = None
        ref_a = arm.historical_ckpt(1000, hist_dir)
        ref_b = arm.historical_ckpt(2000, hist_dir)
        if ref_a.is_file() and ref_b.is_file():
            _, ref_norm, _ = distance(
                flat_tensors(torch.load(ref_a, map_location="cpu", weights_only=False), torch),
                flat_tensors(torch.load(ref_b, map_location="cpu", weights_only=False), torch),
                torch,
            )
            print(f"  reference: ||w(2000) - w(1000)|| = {ref_norm:.6g}  "
                  "(one 1000-step displacement of this same run)")

        for dense_step, hist_step in sorted(SPLICE_PROBES.items()):
            label = f"dense state_dict_{dense_step} vs historical state_dict_{hist_step}"
            if not arm.splice_probe_is_comparable(hist_step):
                msg = (f"{arm.name}: SKIP {label} -- historical run had eval_step="
                       f"{arm.eval_step_in_history}, so a val eval fired before this "
                       "point and consumed global torch RNG. Divergence here is expected "
                       "and is not evidence about the splice.")
                print(f"  {msg}")
                skipped.append(msg)
                continue

            dense_ckpt = dense_dir / "checkpoint" / f"state_dict_{dense_step}.ckpt"
            hist_ckpt = arm.historical_ckpt(hist_step, hist_dir)
            if not dense_ckpt.is_file() or not hist_ckpt.is_file():
                missing = dense_ckpt if not dense_ckpt.is_file() else hist_ckpt
                failures.append(f"{arm.name}: missing {missing}")
                print(f"  MISSING {missing}")
                continue

            max_abs, norm, mismatched = distance(
                flat_tensors(torch.load(dense_ckpt, map_location="cpu", weights_only=False), torch),
                flat_tensors(torch.load(hist_ckpt, map_location="cpu", weights_only=False), torch),
                torch,
            )
            if mismatched:
                print(f"  [WARN] parameters not comparable: {mismatched[:5]}")

            verdict = "?"
            if ref_norm and ref_norm > 0:
                ratio = ref_norm / norm if norm > 0 else float("inf")
                verdict = "SAME TRAJECTORY" if ratio >= SAME_TRAJECTORY_RATIO else "DIVERGED"
                print(f"  [{verdict}] {label}")
                print(f"      max|diff|={max_abs:.6g}  ||diff||={norm:.6g}  "
                      f"reference/diff={ratio:.1f}x  (need >= {SAME_TRAJECTORY_RATIO:.0f}x)")
                if verdict == "DIVERGED":
                    failures.append(f"{arm.name}: {label} -- ||diff||={norm:.6g} vs "
                                    f"reference {ref_norm:.6g}")
            else:
                print(f"  [no reference] {label}: max|diff|={max_abs:.6g} ||diff||={norm:.6g}")

    print()
    print("=" * 78)
    for msg in skipped:
        print(f"SKIPPED: {msg}")
    if failures:
        print(f"SPLICE CHECK FAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        print("\nDo NOT splice. Retrain the affected arm(s) to 40000 with "
              "--checkpoint_steps 100,500,1000,2000,10000,40000 instead.")
        return 1
    print("SPLICE CHECK PASSED -- dense and historical checkpoints lie on one trajectory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

THE NULL MATTERS, AND THE OBVIOUS ONE IS WRONG. This script originally scored the
difference against how far the historical run moved over its own steps 1000->2000, and
demanded 100x smaller. That premise -- "same procedure => near-identical weights" --
assumes training is reproducible, and it is NOT. Measured 2026-07-27 on all8 by running
the same config twice on the same GPU with the same seed:

    step 1001:  ||A - B|| = 546   ||A - historical|| = 579   ||B - historical|| = 554
    step 2001:  ||A - B|| = 1710  ||A - historical|| = 1774  ||B - historical|| = 1698

Two identical runs differ as much as either differs from the historical run. PyG's
scatter-based message passing uses non-deterministic float atomics on CUDA, and training
is chaotic, so a 1e-7 difference at step 1 becomes O(1) in weight space by step 1000. The
old test could not distinguish "same procedure, unreproducible" from "different
procedure" -- which is the only thing it existed to distinguish.

The correct null is the RUN-TO-RUN distance: rerun one arm identically and ask whether
the dense-vs-historical gap is comparable to the dense-vs-replicate gap. If it is, the
historical checkpoints are statistically indistinguishable from a fresh run and the splice
is sound. Without a replicate this script cannot reach a verdict, and it says so rather
than failing closed onto a 5-GPU-hour retrain recommendation.

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

# A dense-vs-historical gap within this factor of the run-to-run (dense-vs-replicate) gap
# is indistinguishable from ordinary irreproducibility. Generous on purpose: the quantity
# it bounds is itself a single noisy sample, so a tight threshold would fail at random.
REPLICATE_TOLERANCE = 3.0

# Prefix of the replicate run used as the null. Produced by rerunning an arm's dense
# config unchanged except for --prefix, e.g.
#   bash train_nm_tucker.sh train_all8_dense.yaml --device 1 --prefix sat_repro_all8
REPLICATE_PREFIX_FMT = "sat_repro_{arm}"


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
    inconclusive: list[str] = []

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

        # The null: an independent rerun of THIS arm's own dense config. Training is not
        # reproducible, so "how far apart are two runs of the same command" is the only
        # meaningful yardstick for "how far apart should the historical run be".
        replicate_dir = resolve_dense_run_dir(
            dense_dir_root, REPLICATE_PREFIX_FMT.format(arm=arm.name))
        if replicate_dir is None:
            print(f"  no replicate ({REPLICATE_PREFIX_FMT.format(arm=arm.name)}_*) -- "
                  "distances will be reported without a verdict")
        else:
            print(f"  replicate={replicate_dir.name}")

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

            # The null: same step, our dense run vs an independent rerun of the same config.
            replicate_norm = None
            if replicate_dir is not None:
                rep_ckpt = replicate_dir / "checkpoint" / f"state_dict_{dense_step}.ckpt"
                if rep_ckpt.is_file():
                    _, replicate_norm, _ = distance(
                        flat_tensors(torch.load(dense_ckpt, map_location="cpu", weights_only=False), torch),
                        flat_tensors(torch.load(rep_ckpt, map_location="cpu", weights_only=False), torch),
                        torch,
                    )

            if replicate_norm is None:
                inconclusive.append(f"{arm.name}: {label} -- no replicate at step {dense_step}")
                print(f"  [INCONCLUSIVE] {label}: ||diff||={norm:.6g}, no run-to-run null "
                      "to compare against")
                continue

            ratio = norm / replicate_norm if replicate_norm > 0 else float("inf")
            ok = ratio <= REPLICATE_TOLERANCE
            print(f"  [{'INDISTINGUISHABLE' if ok else 'DIVERGED'}] {label}")
            print(f"      ||dense - historical|| = {norm:.6g}   max|diff|={max_abs:.6g}")
            print(f"      ||dense - replicate||  = {replicate_norm:.6g}   "
                  f"(two runs of the SAME command)")
            print(f"      ratio = {ratio:.2f}x  (pass if <= {REPLICATE_TOLERANCE:.0f}x)")
            if not ok:
                failures.append(f"{arm.name}: {label} -- {ratio:.2f}x the run-to-run null")

    print()
    print("=" * 78)
    for msg in skipped:
        print(f"SKIPPED: {msg}")
    if failures:
        print(f"SPLICE CHECK FAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        print("\nThe historical checkpoints sit further from a fresh run than two fresh "
              "runs sit from each other, so something really does differ. Do NOT splice; "
              "retrain the affected arm(s) to 40000 with "
              "--checkpoint_steps 100,500,1000,2000,10000,40000.")
        return 1
    if inconclusive:
        print(f"SPLICE CHECK INCONCLUSIVE ({len(inconclusive)}) -- no replicate run to "
              "supply the null. Distances above are unscored; training is NOT reproducible, "
              "so a raw distance means nothing on its own. Produce the null with:")
        for arm in arms:
            print(f"  bash train_nm_tucker.sh {arm.dense_config} --device <gpu> "
                  f"--prefix {REPLICATE_PREFIX_FMT.format(arm=arm.name)}")
        return 2
    print("SPLICE CHECK PASSED -- the dense-vs-historical gap is within the run-to-run "
          "noise of this training setup, i.e. the historical checkpoints are "
          "statistically indistinguishable from a fresh run. Splice is sound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

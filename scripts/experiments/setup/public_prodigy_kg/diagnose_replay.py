"""Native-only repeatability diagnostic; never changes or passes a study gate."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import evaluate_cached as cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "native-eval", "train-run", "failed-quality", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--gpu", type=int, choices=(2, 3), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    paths = {name: cached.native.absolute_path(getattr(args, name), name)
             for name in ("upstream", "native_eval", "train_run", "failed_quality", "output")}
    if paths["output"].exists():
        raise FileExistsError("Diagnostic output must be new")
    cached.native.verify_upstream(paths["upstream"])
    bundle = cached.validate_bundle(paths["native_eval"], paths["train_run"])
    old = cached.read_json(paths["failed_quality"] / "native_replay.json")
    old_protocol = cached.read_json(paths["failed_quality"] / "protocol.json")
    if old_protocol["identity"] != bundle["identity"] or old["replay"]["passed"]:
        raise ValueError("Expected the matching failed native replay")
    first_failure = next(i for i, row in enumerate(old["replay"]["per_episode"])
                         if not row["passed"])
    ordinals = sorted({0, first_failure})
    plan = {"identity": bundle["identity"], "ordinals": ordinals, "repeats": 3,
            "selection": "First episode and first saved replay failure, not outcome selection",
            "purpose": "Native-only numerical repeatability; no scientific arms or gate amendment",
            "dry_run": not args.execute}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    cached.ensure_gpu_idle(args.gpu)
    paths["output"].mkdir(parents=True, exist_ok=False)
    cached.native.write_json(paths["output"] / "protocol.json", plan)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="offline",
                      PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    try:
        torch, _, _ = cached.import_runtime(paths["upstream"])
        numpy = importlib.import_module("numpy")
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
        state = cached.load_state(bundle["checkpoint"], torch)
        model = cached.construct_model(bundle["params"], state, device)
        rng = cached.rng_snapshot(torch, numpy)
        stages = {}
        hooks = []
        names = ("layer_list.0.module_list.0", "layer_list.0.module_list.1",
                 "layer_list.1.proj", "layer_list.2")
        for name, module in model.named_modules():
            if name in names:
                def capture(_m, _a, out, key=name):
                    stages[key] = out.detach().clone().cpu()
                hooks.append(module.register_forward_hook(capture))
        rows = []
        for ordinal in ordinals:
            arguments, reference = next(cached.episode_iterator([bundle["entries"][ordinal]], torch))
            runs = []
            for repeat in range(3):
                model.load_state_dict(state, strict=True)
                model.train(True)
                cached.restore_rng(rng, torch, numpy)
                stages.clear()
                with torch.no_grad():
                    y, logits, _ = model(*cached.clone_arguments(arguments, device))
                runs.append({"y_true": y.detach().clone().cpu(),
                             "logits": logits.detach().clone().cpu(),
                             "stages": dict(stages)})
            comparisons = []
            for repeat, run in enumerate(runs):
                comparisons.append({"repeat": repeat,
                    "against_native": cached.compare_native(run["y_true"], run["logits"], reference),
                    "against_repeat0": cached.compare_native(run["y_true"], run["logits"], runs[0]),
                    "stages_against_repeat0": {name: {
                        "bit_exact": bool(torch.equal(value, runs[0]["stages"][name])),
                        "max_abs_error": float((value.double() - runs[0]["stages"][name].double()).abs().max())}
                        for name, value in run["stages"].items()}})
            rows.append({"ordinal": ordinal, "comparisons": comparisons})
        for hook in hooks:
            hook.remove()
        changed = [name for name, value in model.named_parameters()
                   if not torch.equal(value.detach().cpu(), state[name])]
        if changed:
            raise RuntimeError(f"Weights changed: {changed}")
        cached.native.write_json(paths["output"] / "diagnostic.json", {
            "episodes": rows, "weights_unchanged": True,
            "gate_decision": "None: original replay failure remains unchanged"})
    except BaseException as error:
        cached.native.write_json(paths["output"] / "execution_status.json",
                                 {"status": "failed", "error": str(error)})
        raise
    cached.native.write_json(paths["output"] / "execution_status.json", {"status": "complete"})


if __name__ == "__main__":
    main()

"""Construct exact four-tensor readout interventions; never train or draw weights."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from .verify_member_training import model_digest

READOUT_KEYS = {f"layer_list.0.reset_mlp_{part}.{kind}"
                for part in ("c", "m") for kind in ("weight", "bias")}
MODES = ("trained_background_initial_readout", "initial_background_trained_readout")
POLICIES = {"lowest_sorted", "lowest_shuffled", "uniform_sorted", "uniform_shuffled"}


def hybrid_state(initial, terminal, mode):
    if mode not in MODES or set(initial) != set(terminal) or not READOUT_KEYS <= set(initial):
        raise ValueError("unsupported state layout or intervention")
    if any(initial[k].shape != terminal[k].shape or initial[k].dtype != terminal[k].dtype for k in initial):
        raise ValueError("incompatible parameter tensors")
    # Validate finite source states, including buffers that are not intervened on.
    model_digest(initial)
    model_digest(terminal)
    background, donor = (terminal, initial) if mode == MODES[0] else (initial, terminal)
    result = {k: (donor if k in READOUT_KEYS else background)[k].detach().cpu().clone() for k in background}
    changed = {k for k in result if not torch.equal(result[k], background[k])}
    if not changed or not changed <= READOUT_KEYS:
        raise ValueError("intervention changed nothing or changed unrelated tensors")
    for key in result:
        expected = donor[key] if key in READOUT_KEYS else background[key]
        if not torch.equal(result[key], expected.cpu()):
            raise ValueError("hybrid tensor does not equal its declared donor")
    return result, sorted(changed)


def validate_inventory(receipt, arms):
    gates = ("valid", "research_result", "same_consumed_anchors", "same_final_walk_rng",
             "same_retention_sets_across_role_treatments", "same_initialization_verified")
    expected = {(source, seed, policy) for source in ("ukr_rus", "cp_hk") for seed in range(3) for policy in POLICIES}
    if not all(receipt.get(k) is True for k in gates) or receipt.get("models") != 24 or receipt.get("steps_per_model") != 2500:
        raise ValueError("substantive completed training receipt required")
    if len(arms) != 24 or {(a["source"], a["seed"], a["policy"]) for a in arms} != expected or len({a["model_id"] for a in arms}) != 24:
        raise ValueError("incomplete matched training inventory")
    for seed in range(3):
        if len({a["initial_sha256"] for a in arms if a["seed"] == seed}) != 1:
            raise ValueError("initialization is not shared within seed")
    if len({a["initial_sha256"] for a in arms}) != 3:
        raise ValueError("expected three distinct initializations")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--verified-training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or args.output.exists():
        raise ValueError("invalid thread budget or existing output")
    receipt = json.loads((args.verified_training / "DONE.json").read_text())
    arms = json.loads((args.verified_training / "arms.json").read_text())
    validate_inventory(receipt, arms)
    if args.dry_run:
        print(json.dumps({"models": 48, "training": False, "readout_keys": sorted(READOUT_KEYS), "modes": MODES}))
        return
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True)
    records = []
    for arm in arms:
        terminal_path = Path(arm["checkpoint"])
        initial_path = terminal_path.with_name("state_dict_0.ckpt")
        terminal = torch.load(terminal_path, map_location="cpu", weights_only=True)["model"]
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)["model"]
        if model_digest(initial) != arm["initial_sha256"] or model_digest(terminal) != arm["final_sha256"]:
            raise ValueError("checkpoint no longer matches verified training state")
        for key in READOUT_KEYS:
            if initial[key].shape != ((256, 256) if key.endswith("weight") else (256,)):
                raise ValueError("expected standard 256-dimensional readout")
        for mode in MODES:
            hybrid, changed = hybrid_state(initial, terminal, mode)
            model_id = f"readoutswap_{mode}_{arm['source']}_{arm['policy']}_s{arm['seed']}"
            path = args.output.resolve() / "checkpoints" / f"{model_id}.ckpt"
            path.parent.mkdir(exist_ok=True)
            torch.save({"model": hybrid}, path)
            digest = model_digest(hybrid)
            if model_digest(torch.load(path, map_location="cpu", weights_only=True)["model"]) != digest:
                raise ValueError("saved hybrid does not match constructed tensors")
            records.append({"model_id": model_id, "parent_model_id": arm["model_id"],
                            "source": arm["source"], "sources": [arm["source"]], "seed": arm["seed"],
                            "policy": arm["policy"], "intervention": mode, "checkpoint": str(path),
                            "initial_checkpoint": str(initial_path), "terminal_checkpoint": str(terminal_path),
                            "initial_sha256": arm["initial_sha256"], "terminal_sha256": arm["final_sha256"],
                            "weights_sha256": digest, "changed_keys": changed, "all_other_tensors_identical": True})
    (args.output / "manifest.json").write_text(json.dumps({"models": records, "no_training": True,
        "readout_keys": sorted(READOUT_KEYS), "construction_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()}, indent=2) + "\n")
    (args.output / "model_list.tsv").write_text("model_id\tcheckpoint\tsources\n" + "".join(
        f"{r['model_id']}\t{r['checkpoint']}\t{r['source']}\n" for r in records))
    (args.output / "DONE.json").write_text(json.dumps({"models": len(records), "no_training": True,
        "exact_donor_tensors_verified": True, "only_declared_readout_keys_changed": True}, indent=2) + "\n")
    print(json.dumps({"models": len(records), "no_training": True, "exact_donor_tensors_verified": True}))


if __name__ == "__main__":
    main()

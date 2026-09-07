"""Read-only comparison of earlier/new free-control random states and weights.

Print JSON to stdout. Run on Tucker; historical full input tensors were not saved.
Matching parent RNG/sampler state does not establish matching worker inputs.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch


def state_equal(x, y):
    if type(x) is not type(y):
        return False
    if torch.is_tensor(x):
        return torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return np.array_equal(x, y)
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(state_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(state_equal(v, w) for v, w in zip(x, y))
    return x == y


def compare_consumed(old_params, new_params):
    paths = [Path(p["log_dir"]) / p["exp_name"] / "data/consumed_episodes.jsonl.gz"
             for p in (old_params, new_params)]
    differences, steps, first = {}, 0, None
    with gzip.open(paths[0], "rt") as a, gzip.open(paths[1], "rt") as b:
        for x, y in zip(a, b, strict=True):
            x, y = json.loads(x), json.loads(y)
            steps += 1
            for key in x.keys() | y.keys():
                if x.get(key) != y.get(key):
                    differences[key] = differences.get(key, 0) + 1
                    first = first or steps
    if steps != 2500:
        raise ValueError("incomplete consumed stream")
    return {"model_id": new_params["prefix"], "steps": steps,
            "different_fields_step_counts": differences, "first_difference_step": first}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--current-run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    old = json.loads((args.previous_run / "verified/arms.json").read_text())
    new = json.loads((args.current_run / "verified/arms.json").read_text())
    configs = [{r["prefix"]: r for r in json.loads((p / "manifest.json").read_text())["jobs"]}
               for p in (args.previous_run, args.current_run)]
    controls = [r for r in new if r["condition"] == "free" and r["source"] in {"ukr_rus", "cp_hk"}]
    if len(controls) != 6 or {(r["source"], r["seed"]) for r in controls} != {
            (s, seed) for s in ("ukr_rus", "cp_hk") for seed in range(3)}:
        raise ValueError("complete six-control grid required")
    rows, consumed = [], []
    for r in controls:
        matches = [p for p in old if p["source"] == r["source"] and p["seed"] == r["seed"] and p["policy"] == "lowest_sorted"]
        if len(matches) != 1:
            raise ValueError("previous control not uniquely resolved")
        p = matches[0]
        consumed.append(compare_consumed(configs[0][p["model_id"]], configs[1][r["model_id"]]))
        row = {"model_id": r["model_id"], "previous_model_id": p["model_id"],
               "source": r["source"], "seed": r["seed"], "steps": []}
        for step in (0, 100, 2500):
            xp = Path(p["checkpoint"]).parent / f"training_state_{step}.ckpt"
            yp = Path(r["checkpoint"]).parent / f"training_state_{step}.ckpt"
            x = torch.load(xp, map_location="cpu", weights_only=False)
            y = torch.load(yp, map_location="cpu", weights_only=False)
            tx, ty = x["_training_checkpoint"], y["_training_checkpoint"]
            if x["model"].keys() != y["model"].keys():
                raise ValueError("model layout differs")
            diffs = {k: float((v.double() - y["model"][k].double()).abs().max())
                     for k, v in x["model"].items() if not torch.equal(v, y["model"][k])}
            pars = {k: v for k, v in diffs.items() if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
            row["steps"].append({"step": step, "old_checkpoint": str(xp), "new_checkpoint": str(yp),
                "rng_equal": {k: state_equal(v, ty["rng"][k]) for k, v in tx["rng"].items()},
                "contract_equal": state_equal(tx["parameter_contract"], ty["parameter_contract"]),
                "sampler_equal": state_equal(tx["train_batch_sampler"], ty["train_batch_sampler"]),
                "sampler_components_equal": {k: state_equal(v, ty["train_batch_sampler"][k]) for k, v in tx["train_batch_sampler"].items()},
                "optimizer_equal": state_equal(tx["optimizer"], ty["optimizer"]),
                "model_equal": not diffs, "changed_keys": len(diffs),
                "max_parameter_absolute_difference": max(pars.values(), default=0),
                "largest_tensor_differences": dict(sorted(diffs.items(), key=lambda z: z[1], reverse=True)[:3])})
        rows.append(row)
    print(json.dumps({"read_only": True, "training_pairs": 6, "steps": [0, 100, 2500],
                      "comparisons": rows, "consumed_record_comparisons": consumed}))


if __name__ == "__main__":
    main()

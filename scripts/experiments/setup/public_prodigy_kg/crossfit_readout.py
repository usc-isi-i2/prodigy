"""Query-role cross-fitted supports: fixed first 64 discovery episodes."""
import argparse
import json
import os
from pathlib import Path

from . import run_native as native


def hide_support(arguments, held_rows):
    """Remove held labels from every supplied label-bearing input."""
    import torch
    changed = [v.clone() for v in arguments]
    n, ways = changed[2].shape
    mask = changed[5].reshape(n, ways)
    if mask[held_rows].any():
        raise ValueError("Only original supports may be held out")
    mask[held_rows] = True
    attrs = changed[4].reshape(n, ways, -1)
    if attrs.shape[-1] != 2:
        raise ValueError("Expected native two-field metagraph edge attributes")
    attrs[held_rows, :, 0] = 1
    attrs[held_rows, :, 1] = 0
    changed[2][held_rows] = 0
    # Pinned MetaGNN ignores sequences, but do not leave labels in unused inputs.
    for i in range(6, len(changed)):
        changed[i] = torch.zeros_like(changed[i])
    return tuple(changed)


def crossfit(model, artifact, device):
    import torch
    from .readout_episode import run_episode
    from .episode_mechanism import capture_decoder
    from .paired_replay import isolated_forward_state
    from .role_centering import predict
    from .score_readout import score_task

    meta = [m for m in model.layer_list if hasattr(m, "gnn_layers")]
    if len(meta) != 1 or type(meta[0]).__name__ != "MetaGNN":
        raise ValueError("Only the audited non-sequence MetaGNN is supported")
    base = run_episode(model, artifact, device, atol=1e-4, compare_post=True)
    if len(base["tasks"]) != 1:
        raise ValueError("Expected one task")
    task = base["tasks"][0]
    support, query = task["support_rows"], task["query_rows"]
    labels = artifact["input"][2]
    groups = [support[labels[support].argmax(1) == c] for c in range(labels.shape[1])]
    if labels.shape[1] != 20 or any(len(g) != 3 for g in groups):
        raise ValueError("Expected 20-way, 3-shot support")
    features = base["geometry"]["post"].clone()
    folds = []
    for fold in range(3):
        held = torch.stack([g[fold] for g in groups])
        arguments = tuple(v.to(device) for v in hide_support(artifact["input"], held))
        with isolated_forward_state(model, artifact["state"]), torch.no_grad(), capture_decoder(model) as decoded:
            model(*arguments)
        features[held] = decoded[0]["input_x"][held.to(device)].cpu()
        folds.append(held.tolist())
    truth = labels[query].argmax(1).numpy()
    native_logits = base["native"]["logits"].numpy()
    scores, logits = {}, {}
    for name, x in (("pre", base["geometry"]["pre"]), ("post", base["geometry"]["post"]),
                    ("crossfit", features)):
        for center in ("none", "support"):
            z = predict(x[support], labels[support], x[query], center)
            key = name + "/" + center
            logits[key] = z
            scores[key] = score_task(truth, native_logits, z.numpy())
    return {"scores": scores, "logits": logits, "folds": folds, "query_labels": labels[query],
            "support_rows": support, "query_rows": query,
            "native_parity": base["native"]["max_abs_error"],
            "query_representation_unchanged": torch.equal(features[query], base["geometry"]["post"][query])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for name in ("source", "checkpoint", "upstream", "output"):
        setattr(args, name, getattr(args, name).resolve())
    from .run_saved_readout import inspect_run, construct_model
    protocol, records = inspect_run(args.source, args.checkpoint)
    native.verify_upstream(args.upstream)
    plan = {"source": str(args.source), "checkpoint_sha256": protocol["checkpoint_sha256"],
            "episodes": 64, "selection": "first64 discovery episodes; not the frozen fresh stream",
            "folds": "one held support per class in each of three passes",
            "query_pass": "unchanged native full-support query embeddings",
            "confound": "cross-fitted supports see two labels per class; native queries see three",
            "rules": "ridge lambda1, scale1, row-L2; none/support centering only",
            "scope": "mechanism pilot, not deployment benchmark; four forward passes"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="offline")
    native.runtime_versions()
    native.isolate_native_path(args.upstream)
    os.chdir(args.upstream)
    import torch
    import numpy as np
    torch.set_num_threads(1)
    params = json.loads((args.source / "effective_params.json").read_text())
    model = construct_model(params, args.checkpoint, "cuda:0")
    native.verify_imports(args.upstream)
    args.output.mkdir(parents=True, exist_ok=False)
    native.write_json(args.output / "protocol.json", plan)
    rows = []
    try:
        for record in records[:64]:
            path = args.source / "paired_episodes" / record["file"]
            if native.file_sha256(path) != record["sha256"]:
                raise ValueError("Native artifact hash mismatch")
            result = crossfit(model, torch.load(path, map_location="cpu")["native_capture"], "cuda:0")
            torch.save({"source_sha256": record["sha256"], **result}, args.output / record["file"])
            rows.append(result["scores"])
            print(f"Cross-fit episode {len(rows)}/64", flush=True)
        means = {name: {metric: float(np.mean([r[name]["metrics"]["ridge"][metric] for r in rows]))
                        for metric in rows[0][name]["metrics"]["ridge"]} for name in rows[0]}
        means["native"] = {m: float(np.mean([r["pre/none"]["metrics"]["native"][m] for r in rows]))
                           for m in rows[0]["pre/none"]["metrics"]["native"]}
        native.write_json(args.output / "summary.json", {"episodes": len(rows), "means": means})
    except BaseException as error:
        native.write_json(args.output / "execution_status.json", {"status": "failed", "error": repr(error)})
        raise
    native.write_json(args.output / "execution_status.json", {"status": "complete"})


if __name__ == "__main__":
    main()

"""Replay the completed public mechanism stream with a support-only readout.

Dry-run by default. Constructs only the pinned native model: no graph loading,
new episodes, training, W&B, or modification of the source run.
"""
import argparse
import json
import os
from pathlib import Path

from . import run_native as native


def inspect_run(run, checkpoint):
    protocol = json.loads((run / "protocol.json").read_text())
    status = json.loads((run / "execution_status.json").read_text())
    records = json.loads((run / "paired_episodes/index.json").read_text())["records"]
    if status["status"] != "complete" or len(records) != protocol["paired_mechanism"]["episode_count"]:
        raise ValueError("Source must be a complete native stream")
    if [r["ordinal"] for r in records] != list(range(len(records))) or not records:
        raise ValueError("Invalid source episode inventory")
    if native.file_sha256(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from saved native run")
    for record in records:
        path = run / "paired_episodes" / record["file"]
        if path.resolve().parent != (run / "paired_episodes").resolve():
            raise ValueError("Episode path escapes source directory")
    return protocol, records


def construct_model(params, checkpoint, device):
    # Matches pinned upstream TrainerFS.__init__ model construction, without its
    # dataset, sentence encoder, optimizer, or logging side effects.
    import torch
    from experiments.layers import get_module_list
    from models.general_gnn import SingleLayerGeneralGNN

    if params["dataset"] != "FB15K-237" or params["input_dim"] != 770 or params["kg_emb_model"]:
        raise ValueError("This constructor is restricted to the nominated public recipe")
    if params["layers"] != "S2,UX,M2":
        raise ValueError("Unexpected public architecture")
    label_mlp = torch.nn.Linear(768, params["emb_dim"])
    layers = get_module_list(params["layers"], params["emb_dim"], edge_attr_dim=768,
        input_dim=params["input_dim"], dropout=params["dropout"],
        reset_after_layer=params["reset_after_layer"], attention_mask_scheme=params["attention_mask_scheme"],
        has_final_back=params["has_final_back"], msg_pos_only=params.get("meta_gnn_pos_only", False),
        batch_norm_metagraph=not params["no_bn_metagraph"],
        batch_norm_encoder=not params["no_bn_encoder"], gnn_use_relu=True)
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(layers), initial_label_mlp=label_mlp,
        params=params, text_dropout=torch.nn.Dropout(params["text_features_dropout"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=True)
    return model.to(device)  # Per-episode replay restores captured modes and BN.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "checkpoint", "upstream", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for field in ("run", "checkpoint", "upstream", "output"):
        setattr(args, field, getattr(args, field).resolve())
    protocol, records = inspect_run(args.run, args.checkpoint)
    verification = native.verify_upstream(args.upstream)
    plan = {"source_run": str(args.run.resolve()), "episodes": len(records),
            "checkpoint_sha256": protocol["checkpoint_sha256"], "upstream": verification,
            "source_protocol_sha256": native.file_sha256(args.run / "protocol.json"),
            "source_index_sha256": native.file_sha256(args.run / "paired_episodes/index.json"),
            "physical_gpu": args.gpu, "dry_run": not args.execute,
            "parity": protocol["paired_mechanism"],
            "readout": "row-L2 ridge, lambda=1, no intercept, scale=1, support labels only"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="offline")
    native.runtime_versions()
    native.isolate_native_path(args.upstream.resolve())
    # Upstream modules append relative entries to sys.path. Match native.execute
    # so these resolve within upstream, never back into the wrapper checkout.
    os.chdir(args.upstream)
    import torch
    import numpy as np
    from .readout_episode import run_episode
    from .score_readout import score_task
    from scripts.experiments.setup.trace_health_guided.multiclass_metrics import paired_summary

    params = json.loads((args.run / "effective_params.json").read_text())
    if params["batch_size"] != 1 or params["n_way"] != 20 or params["n_query"] != 4:
        raise ValueError("Summary expects the nominated single-task 20-way, 4-query stream")
    model = construct_model(params, args.checkpoint, "cuda:0")
    native.verify_imports(args.upstream.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    native.write_json(args.output / "protocol.json", plan)
    native_predictions, ridge_predictions, metrics = {}, {}, []
    try:
        for record in records:
            path = args.run / "paired_episodes" / record["file"]
            if native.file_sha256(path) != record["sha256"]:
                raise ValueError("Saved episode hash mismatch")
            artifact = torch.load(path, map_location="cpu")["native_capture"]
            result = run_episode(model, artifact, "cuda:0",
                atol=protocol["paired_mechanism"]["parity_atol"],
                rtol=protocol["paired_mechanism"]["parity_rtol"])
            if len(result["tasks"]) != 1 or result["native"]["logits"].shape != (80, 20):
                raise ValueError("Episode differs from nominated task/query inventory")
            scored = []
            for task, group in enumerate(result["tasks"]):
                q = group["query_indices"]
                onehot = result["native"]["y_true"][q]
                if not torch.all((onehot == 0) | (onehot == 1)) or not torch.all(onehot.sum(1) == 1):
                    raise ValueError("Expected one-hot native query labels")
                labels = onehot.argmax(1).numpy()
                full, ridge = result["native"]["logits"][q].numpy(), result["ridge_logits"][q].numpy()
                key = (record["ordinal"], task)
                native_predictions[key], ridge_predictions[key] = (labels, full), (labels, ridge)
                score = score_task(labels, full, ridge)
                scored.append(score)
                metrics.append(score["metrics"])
            torch.save({"source_sha256": record["sha256"], "result": result, "scores": scored},
                       args.output / record["file"])
            print(f"Scored saved episode {record['ordinal'] + 1}/{len(records)}", flush=True)
        summary = {"episodes": len(records), "tasks": len(metrics),
                   "means": {arm: {metric: float(np.mean([m[arm][metric] for m in metrics]))
                                    for metric in metrics[0][arm]} for arm in ("native", "ridge")},
                   "paired_deltas": paired_summary(native_predictions, ridge_predictions),
                   "uncertainty_scope": "Conditional on this checkpoint and target; recurring entities can induce dependence."}
        native.write_json(args.output / "summary.json", summary)
    except BaseException as error:
        native.write_json(args.output / "execution_status.json", {"status": "failed", "error": repr(error)})
        raise
    native.write_json(args.output / "execution_status.json", {"status": "complete"})


if __name__ == "__main__":
    main()

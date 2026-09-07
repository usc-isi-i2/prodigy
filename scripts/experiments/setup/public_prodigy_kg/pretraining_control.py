"""Saved-initialization and endpoint-text controls for public readout gains."""
import argparse
import json
import os
from pathlib import Path

from . import run_native as native


def extract_pre(model, artifact, device):
    import torch
    from .paired_replay import isolated_forward_state

    class Captured(Exception):
        pass
    meta = [m for m in model.layer_list if hasattr(m, "gnn_layers")][0]
    captured = []
    def before(_module, _args, kwargs):
        captured.append(kwargs["x"][:kwargs["start_right"]].detach().clone())
        raise Captured()
    # Keep the initialization's buffers; use saved episode modes and RNG only.
    state = dict(artifact["state"])
    state["buffers"] = {n: b.detach().clone().cpu() for n, b in model.named_buffers()}
    handle = meta.register_forward_pre_hook(before, with_kwargs=True)
    try:
        with isolated_forward_state(model, state), torch.no_grad():
            try:
                model(*(v.clone().to(device) for v in artifact["input"]))
            except Captured:
                pass
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Did not capture exactly one pre-metagraph embedding")
    return captured[0].cpu()


def endpoint_features(graph):
    import torch
    if graph.x.ndim != 2 or graph.x.shape[1] != 770 or not torch.all(graph.ptr[1:] - graph.ptr[:-1] >= 3):
        raise ValueError("Expected native public 768d text plus two flags")
    head, tail = graph.ptr[:-1], graph.ptr[:-1] + 1
    # Upstream pooling explicitly identifies the first two nodes as endpoints.
    expected = torch.stack((head, tail), 1).flatten()
    if not torch.equal(graph.edge_index_supernode[0], expected):
        raise ValueError("Endpoint order differs from native pooling edges")
    return torch.cat((graph.x[head, :768], graph.x[tail, :768]), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "stage-run", "initialization", "upstream", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    for root in (args.source, args.stage_run):
        if json.loads((root / "execution_status.json").read_text())["status"] != "complete":
            raise ValueError("Source runs must be complete")
    stage_protocol = json.loads((args.stage_run / "protocol.json").read_text())
    index = args.source / "paired_episodes/index.json"
    if native.file_sha256(index) != stage_protocol["source_index_sha256"]:
        raise ValueError("Stage and native source differ")
    records = json.loads(index.read_text())["records"]
    if len(records) != 500 or [r["ordinal"] for r in records] != list(range(500)):
        raise ValueError("Expected complete 500 discovery episodes")
    native.verify_upstream(args.upstream)
    plan = {"source": str(args.source), "stage_run": str(args.stage_run),
            "initialization": str(args.initialization), "initialization_sha256": native.file_sha256(args.initialization),
            "source_index_sha256": native.file_sha256(index), "episodes": 500,
            "features": ["trained pre-metagraph", "saved-initialization pre-metagraph", "head-tail text concatenation"],
            "readouts": "ridge lambda1, scale1, no intercept; none/support centering then row-L2",
            "scope": "GFM pretraining control; text embeddings are pretrained in every arm"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="offline")
    native.runtime_versions()
    native.isolate_native_path(args.upstream)
    os.chdir(args.upstream)
    import torch
    import numpy as np
    from .run_saved_readout import construct_model
    from .role_centering import predict
    from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics
    torch.set_num_threads(1)
    params = json.loads((args.source / "effective_params.json").read_text())
    model = construct_model(params, args.initialization, "cuda:0")
    native.verify_imports(args.upstream)
    args.output.mkdir(parents=True, exist_ok=False)
    native.write_json(args.output / "protocol.json", plan)
    rows = []
    try:
        for record in records:
            path = args.source / "paired_episodes" / record["file"]
            if path.resolve().parent != (args.source / "paired_episodes").resolve() or native.file_sha256(path) != record["sha256"]:
                raise ValueError("Source artifact receipt failed")
            artifact = torch.load(path, map_location="cpu")["native_capture"]
            stage = torch.load(args.stage_run / record["file"], map_location="cpu")
            if stage["source_sha256"] != record["sha256"]:
                raise ValueError("Stage artifact source mismatch")
            result = stage["result"]
            group = result["tasks"][0]
            s, q = group["support_rows"], group["query_rows"]
            labels = artifact["input"][2]
            if not torch.equal(labels[q], result["native"]["y_true"]):
                raise ValueError("Query label ordering differs")
            truth = labels[q].argmax(1).numpy()
            features = {"trained": result["geometry"]["pre"],
                        "initialization": extract_pre(model, artifact, "cuda:0"),
                        "endpoint_text": endpoint_features(artifact["input"][0])}
            metrics = {"native": episode_metrics(truth, result["native"]["logits"].numpy())}
            outputs = {}
            for name, x in features.items():
                for center in ("none", "support"):
                    key = name + "/" + center
                    z = predict(x[s], labels[s], x[q], center)
                    if key == "trained/none" and not torch.allclose(z, result["ridge_logits"], atol=1e-5, rtol=0):
                        raise ValueError("Trained readout baseline parity failed")
                    metrics[key] = episode_metrics(truth, z.numpy())
                    outputs[key] = z
            rows.append(metrics)
            torch.save({"source_sha256": record["sha256"], "metrics": metrics, "logits": outputs,
                        "query_labels": labels[q]}, args.output / record["file"])
            if len(rows) % 50 == 0:
                print(f"Pretraining control {len(rows)}/500", flush=True)
        means = {name: {m: float(np.mean([row[name][m] for row in rows])) for m in rows[0][name]} for name in rows[0]}
        native.write_json(args.output / "summary.json", {"episodes": len(rows), "means": means})
    except BaseException as error:
        native.write_json(args.output / "execution_status.json", {"status": "failed", "error": repr(error)})
        raise
    native.write_json(args.output / "execution_status.json", {"status": "complete"})


if __name__ == "__main__":
    main()

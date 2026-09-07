"""Fixed 2k/4k/8k checkpoint readouts on the first128 saved fresh episodes."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from . import run_native as native

STEPS = (2000, 4000, 8000)
EPISODES = 128


def nominated(checkpoint_dir):
    return {step: checkpoint_dir / f"state_dict_{step}.ckpt" for step in STEPS}


def forward_checkpoint(model, artifact, device):
    """Shared saved inputs/RNG/modes, but ONLY the loaded checkpoint's buffers."""
    import torch
    from .paired_replay import isolated_forward_state
    state = dict(artifact["state"])
    state["buffers"] = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    meta = [m for m in model.layer_list if hasattr(m, "gnn_layers")][0]
    captured = []
    def capture(_module, _args, kwargs):
        captured.append(kwargs["x"][:kwargs["start_right"]].detach().cpu().clone())
    handle = meta.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with isolated_forward_state(model, state), torch.no_grad():
            yt, logits, *_ = model(*(value.clone().to(device) for value in artifact["input"]))
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one actual pre-metagraph stage")
    return yt.detach().cpu(), logits.detach().cpu(), captured[0]


def main():
    p = argparse.ArgumentParser(__doc__)
    for field in ("source", "checkpoint-dir", "upstream", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--gpu", type=int, choices=(2,3), required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    checkpoints = nominated(args.checkpoint_dir)
    source_protocol = json.loads((args.source / "protocol.json").read_text())
    records = json.loads((args.source / "paired_episodes/index.json").read_text())["records"]
    if json.loads((args.source / "execution_status.json").read_text())["status"] != "complete" or len(records) != 500 or [r["ordinal"] for r in records] != list(range(500)):
        raise ValueError("Complete nominated500 episode stream required")
    if native.file_sha256(checkpoints[8000]) != source_protocol["checkpoint_sha256"]:
        raise ValueError("8k checkpoint differs from saved stream")
    verification = native.verify_upstream(args.upstream)
    plan = dict(steps=list(STEPS), episodes=EPISODES, source=str(args.source),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        effective_params_sha256=native.file_sha256(args.source / "effective_params.json"),
        checkpoint_sha256={str(step):native.file_sha256(path) for step,path in checkpoints.items()},
        checkpoints={str(step):str(path) for step,path in checkpoints.items()},
        source_index_sha256=native.file_sha256(args.source / "paired_episodes/index.json"),
        source_protocol_sha256=native.file_sha256(args.source / "protocol.json"), upstream=verification,
        buffers="Each checkpoint's own buffers reset before every episode; never use captured8k buffers",
        shared="Exact saved inputs, per-module train modes and RNG across checkpoints",
        readout="support-centered rowL2 ridge lambda1 scale1 nointercept", parity_8k_atol=1e-4,
        selection="Retrospective fixed trajectory, no checkpoint selection claim")
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
    if any(params[k] != v for k,v in dict(batch_size=1,n_way=20,n_shots=3,n_query=4).items()):
        raise ValueError("Expected fixed single-task 20-way 3-shot 4-query protocol")
    args.output.mkdir(parents=True, exist_ok=False)
    native.write_json(args.output / "protocol.json", plan)
    native.write_json(args.output / "execution_status.json", {"status":"running"})
    rows, receipts = [], []
    try:
        for step,path in checkpoints.items():
            model = construct_model(params, path, "cuda:0")
            native.verify_imports(args.upstream)
            directory = args.output / str(step)
            directory.mkdir()
            for record in records[:EPISODES]:
                source = args.source / "paired_episodes" / record["file"]
                if source.resolve().parent != (args.source / "paired_episodes").resolve() or native.file_sha256(source) != record["sha256"]:
                    raise ValueError("Source episode receipt failed")
                saved = torch.load(source, map_location="cpu")
                artifact, ref = saved["native_capture"], saved["result"]
                if not all(artifact["state"]["training"].values()):
                    raise ValueError("Expected native public train-mode forward")
                yt, logits, x = forward_checkpoint(model, artifact, "cuda:0")
                if len(ref["tasks"]) != 1:
                    raise ValueError("Expected single-task episode")
                group = ref["tasks"][0]
                s,q = group["support_rows"], group["query_rows"]
                labels = artifact["input"][2]
                if not torch.equal(yt, labels[q]):
                    raise ValueError("Query labels/order differ")
                ridge = predict(x[s], labels[s], x[q], "support")
                if logits.shape != (80,20) or ridge.shape != logits.shape or not torch.isfinite(logits).all() or not torch.isfinite(ridge).all():
                    raise ValueError("Invalid nominated query scores")
                if step == 8000:
                    torch.testing.assert_close(logits, ref["native"]["logits"], atol=1e-4, rtol=0)
                    torch.testing.assert_close(ridge, ref["support_centered_logits"], atol=1e-4, rtol=0)
                truth = yt.argmax(1).numpy()
                metrics = {"native":episode_metrics(truth, logits.numpy()), "centered":episode_metrics(truth,ridge.numpy())}
                rows.append(dict(step=step, ordinal=record["ordinal"], metrics=metrics))
                destination = directory / record["file"]
                torch.save(dict(source_sha256=record["sha256"], y_true_onehot=yt, native_logits=logits,
                    centered_logits=ridge, pre_embeddings=x, metrics=metrics), destination)
                receipts.append(dict(step=step, file=str(destination.relative_to(args.output)), sha256=native.file_sha256(destination), source_sha256=record["sha256"]))
                if (record["ordinal"]+1)%16 == 0:
                    print(f"step{step}: {record['ordinal']+1}/{EPISODES}",flush=True)
            del model
        means = {str(step):{arm:{metric:float(np.mean([r["metrics"][arm][metric] for r in rows if r["step"]==step])) for metric in rows[0]["metrics"][arm]} for arm in ("native","centered")} for step in STEPS}
        native.write_json(args.output / "summary.json", dict(rows=rows, means=means, receipts=receipts))
        native.write_json(args.output / "execution_status.json", dict(status="complete"))
    except BaseException as error:
        native.write_json(args.output / "execution_status.json", dict(status="failed",error=repr(error)))
        raise


if __name__ == "__main__":
    main()

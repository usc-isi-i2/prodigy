"""Saved 2k/8k support/query donors through fixed native 8k inference.

Dry run unless --execute. This is a pre-M role crossover, not a fixed-final-query
intervention: native training-mode BN and M2 can couple support/query/label rows.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess

from . import run_native as native

CONDITIONS = ((2000, 2000), (2000, 8000), (8000, 2000), (8000, 8000))
EPISODES = 128
ATOL = 1e-4


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assemble_rows(support_donor, query_donor, queries):
    import torch
    if support_donor.ndim != 2 or support_donor.shape != query_donor.shape:
        raise ValueError("Donor shapes differ or are not matrices")
    if support_donor.dtype != query_donor.dtype or support_donor.device != query_donor.device:
        raise ValueError("Donor dtype/device differs")
    if not support_donor.is_floating_point() or not torch.isfinite(support_donor).all() or not torch.isfinite(query_donor).all():
        raise ValueError("Donors must be finite floating point matrices")
    if queries.dtype != torch.bool or queries.shape != support_donor.shape[:1] or queries.device != support_donor.device:
        raise ValueError("Expected explicit aligned boolean query mask")
    if not queries.any() or queries.all():
        raise ValueError("Both support and query roles required")
    result = support_donor.detach().clone()
    result[queries] = query_donor[queries]
    return result


def validate_roles(arguments, task):
    import torch
    from .role_interventions import query_rows
    queries = query_rows(arguments)
    for key, mask in (("support_rows", ~queries), ("query_rows", queries)):
        expected = mask.nonzero(as_tuple=True)[0].cpu()
        actual = torch.as_tensor(task[key]).cpu()
        if actual.dtype not in (torch.int32, torch.int64) or not torch.equal(actual, expected):
            raise ValueError("Saved task ordering differs: " + key)
    return queries


@contextmanager
def clamp_roles(model, natural_late, replacement):
    """Check natural late U1, then replace data only before native M2."""
    import torch
    metas = [module for module in model.layer_list if hasattr(module, "gnn_layers")]
    if len(metas) != 1:
        raise ValueError("Expected one metagraph module")
    audit = {"calls": 0}
    def replace(_module, positional, kwargs):
        if audit["calls"]:
            raise ValueError("Repeated metagraph invocation")
        x, boundary = kwargs["x"], kwargs["start_right"]
        if natural_late.shape != replacement.shape or x[:boundary].shape != natural_late.shape:
            raise ValueError("Pre-M donor boundary/shape mismatch")
        if x.dtype != natural_late.dtype or x.dtype != replacement.dtype:
            raise ValueError("Pre-M donor dtype mismatch")
        expected = natural_late.to(x.device)
        if not torch.isfinite(x).all() or not torch.allclose(x[:boundary], expected, atol=ATOL, rtol=0):
            raise ValueError("Natural late U1 numerical parity failed")
        changed = x.clone()
        changed[:boundary] = replacement.to(x.device)
        if not torch.equal(changed[boundary:], x[boundary:]):
            raise ValueError("Label rows changed at pre-M hook")
        audit.update(calls=1, natural_max_abs_drift=float((x[:boundary]-expected).abs().max()),
                     data_rows=int(boundary), label_rows=int(x.shape[0]-boundary),
                     label_rows_sha256=tensor_hash(x[boundary:]),
                     replacement_sha256=tensor_hash(changed[:boundary]))
        return positional, dict(kwargs, x=changed)
    handle = metas[0].register_forward_pre_hook(replace, with_kwargs=True)
    try:
        yield audit
        if audit["calls"] != 1:
            raise ValueError("Missing pre-M hook invocation")
    finally:
        handle.remove()


def receipt_path(root, receipt):
    path = (root / receipt["file"]).resolve()
    if not path.is_relative_to(root.resolve()) or native.file_sha256(path) != receipt["sha256"]:
        raise ValueError("Artifact path/hash receipt mismatch")
    return path


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("source", "trajectory", "crossover", "checkpoint-dir", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output.exists():
        raise ValueError("Output already exists; never overwrite")
    for root in (args.source, args.trajectory, args.crossover):
        if json.loads((root / "execution_status.json").read_text())["status"] != "complete":
            raise ValueError("Completed source, trajectory and crossover required")
    records = json.loads((args.source / "paired_episodes/index.json").read_text())["records"]
    if len(records) != 500 or [r["ordinal"] for r in records] != list(range(500)):
        raise ValueError("Expected complete ordered 500-episode source")
    index_hash = native.file_sha256(args.source / "paired_episodes/index.json")
    param_hash = native.file_sha256(args.source / "effective_params.json")
    hashes = {str(step): native.file_sha256(args.checkpoint_dir / f"state_dict_{step}.ckpt") for step in (2000, 8000)}
    protocols = {name: json.loads((getattr(args, name) / "protocol.json").read_text()) for name in ("trajectory", "crossover")}
    for protocol in protocols.values():
        if protocol["source_index_sha256"] != index_hash or protocol["effective_params_sha256"] != param_hash:
            raise ValueError("Source/parameter provenance mismatch")
        if any(protocol["checkpoint_sha256"][step] != value for step, value in hashes.items()):
            raise ValueError("Checkpoint provenance mismatch")
    if protocols["crossover"]["trajectory_summary_sha256"] != native.file_sha256(args.trajectory / "summary.json"):
        raise ValueError("Crossover trajectory summary changed")
    if json.loads((args.source / "protocol.json").read_text())["checkpoint_sha256"] != hashes["8000"]:
        raise ValueError("Source late checkpoint mismatch")
    plan = dict(conditions=[list(c) for c in CONDITIONS], episodes=EPISODES, inference_step=8000,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                checkpoint_sha256=hashes, source_index_sha256=index_hash, effective_params_sha256=param_hash,
                roots={name:str(getattr(args,name)) for name in ("source","trajectory","crossover")},
                receipt_sha256={name:{file:native.file_sha256(getattr(args,name)/file) for file in ("protocol.json","summary.json")} for name in ("trajectory","crossover")},
                source_protocol_sha256=native.file_sha256(args.source / "protocol.json"),
                upstream=native.verify_upstream(args.upstream), parity_atol=ATOL, parity_rtol=0,
                endpoints="S2Q2 must reproduce E2000I8000; S8Q8 must reproduce E8000I8000",
                primary_metric="accuracy",
                prediction="Late support donors reduce accuracy at both fixed query endpoints; compare query costs without presuming support dominance; retain null/opposite outcomes",
                contract="Exact saved pre-M data donors; natural late U1 parity first; late label rows untouched",
                scope="Native train-mode BN/M2: pre-M query preservation is NOT final-query preservation",
                fitting=False, selection=False)
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
    from .run_trajectory import forward_checkpoint
    from .run_checkpoint_crossover import partition
    from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics
    torch.set_num_threads(1)
    params = json.loads((args.source / "effective_params.json").read_text())
    if tuple(params[k] for k in ("layers","batch_size","n_way","n_shots","n_query")) != ("S2,UX,M2",1,20,3,4):
        raise ValueError("Unexpected native model/episode protocol")
    checkpoint = args.checkpoint_dir / "state_dict_8000.ckpt"
    model = construct_model(params, checkpoint, "cuda:0")
    native.verify_imports(args.upstream)
    frozen = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    plan["state_partition"] = partition(frozen)
    trajectory = {(r["step"], r["file"]):r for r in json.loads((args.trajectory / "summary.json").read_text())["receipts"]}
    crossover = {r["file"]:r for r in json.loads((args.crossover / "summary.json").read_text())["receipts"]}
    args.output.mkdir(parents=True, exist_ok=False)
    native.write_json(args.output / "protocol.json", plan)
    native.write_json(args.output / "execution_status.json", dict(status="running"))
    rows, receipts = [], []
    try:
        for record in records[:EPISODES]:
            source = receipt_path(args.source / "paired_episodes", record)
            saved = torch.load(source, map_location="cpu")
            artifact, result = saved["native_capture"], saved["result"]
            if not all(artifact["state"]["training"].values()) or len(result["tasks"]) != 1:
                raise ValueError("Native train-mode single-task capture required")
            queries = validate_roles(artifact["input"], result["tasks"][0])
            if int(queries.sum()) != 80 or int((~queries).sum()) != 60:
                raise ValueError("Unexpected role counts")
            donors, donor_receipts, endpoints = {}, {}, {}
            for step in (2000,8000):
                receipt = trajectory[(step, f"{step}/{record['file']}")]
                if receipt["source_sha256"] != record["sha256"]:
                    raise ValueError("Trajectory episode source mismatch")
                donors[step] = torch.load(receipt_path(args.trajectory, receipt), map_location="cpu")
                donor_receipts[step] = receipt
                endpoint = crossover[f"E{step}I8000/{record['file']}"]
                if endpoint["source_sha256"] != record["sha256"]:
                    raise ValueError("Crossover episode source mismatch")
                endpoints[step] = torch.load(receipt_path(args.crossover, endpoint), map_location="cpu")
                if endpoints[step]["trajectory_sha256"] != receipt["sha256"] or donors[step]["source_sha256"] != record["sha256"]:
                    raise ValueError("Endpoint donor receipt mismatch")
                if not torch.equal(donors[step]["y_true_onehot"], artifact["input"][2][queries]):
                    raise ValueError("Donor query labels/order mismatch")
            for support, query in CONDITIONS:
                name = f"S{support}Q{query}I8000"
                replacement = assemble_rows(donors[support]["pre_embeddings"], donors[query]["pre_embeddings"], queries)
                with clamp_roles(model, donors[8000]["pre_embeddings"], replacement) as audit:
                    truth, logits, captured = forward_checkpoint(model, artifact, "cuda:0")
                if not torch.equal(captured, replacement) or not torch.equal(truth, donors[8000]["y_true_onehot"]):
                    raise ValueError("Exact clamped rows/query labels mismatch")
                if logits.shape != (80,20) or not torch.isfinite(logits).all():
                    raise ValueError("Invalid output logits")
                endpoint_error = None
                if support == query:
                    expected = endpoints[support]
                    if not torch.equal(truth, expected["y_true_onehot"]):
                        raise ValueError("Endpoint query labels mismatch")
                    torch.testing.assert_close(logits, expected["logits"], atol=ATOL, rtol=0)
                    endpoint_error = float((logits-expected["logits"]).abs().max())
                for key, value in model.state_dict().items():
                    if not torch.equal(value.cpu(), frozen[key]):
                        raise ValueError("Late checkpoint state changed: " + key)
                metrics = episode_metrics(truth.argmax(1).numpy(), logits.numpy())
                row = dict(condition=name, support_step=support, query_step=query, ordinal=record["ordinal"],
                           metrics=metrics, clamp=audit, endpoint_max_abs_error=endpoint_error)
                rows.append(row)
                directory = args.output / name
                directory.mkdir(exist_ok=True)
                out = directory / record["file"]
                if out.exists():
                    raise ValueError("Duplicate output episode")
                torch.save(dict(row=row, logits=logits, y_true_onehot=truth, source_sha256=record["sha256"],
                                query_mask=queries.clone(), support_mask=(~queries).clone(),
                                donor_sha256={str(s):donor_receipts[s]["sha256"] for s in donors},
                                donor_tensor_sha256={str(s):tensor_hash(donors[s]["pre_embeddings"]) for s in donors}), out)
                receipts.append(dict(file=str(out.relative_to(args.output)), sha256=native.file_sha256(out), source_sha256=record["sha256"]))
            if (record["ordinal"]+1)%16 == 0:
                print(f"All four roles: {record['ordinal']+1}/{EPISODES}", flush=True)
        means = {f"S{s}Q{q}I8000":{m:float(np.mean([r["metrics"][m] for r in rows if (r["support_step"],r["query_step"])==(s,q)])) for m in rows[0]["metrics"]} for s,q in CONDITIONS}
        effects = {m:dict(support_given_early_query=means["S8000Q2000I8000"][m]-means["S2000Q2000I8000"][m],
                         support_given_late_query=means["S8000Q8000I8000"][m]-means["S2000Q8000I8000"][m],
                         query_given_early_support=means["S2000Q8000I8000"][m]-means["S2000Q2000I8000"][m],
                         query_given_late_support=means["S8000Q8000I8000"][m]-means["S8000Q2000I8000"][m],
                         interaction=means["S8000Q8000I8000"][m]-means["S2000Q8000I8000"][m]-means["S8000Q2000I8000"][m]+means["S2000Q2000I8000"][m]) for m in rows[0]["metrics"]}
        native.write_json(args.output / "summary.json", dict(rows=rows, means=means, conditional_effects=effects, receipts=receipts, interpretation="Conditional contrasts, not additive mediation"))
        native.write_json(args.output / "execution_status.json", dict(status="complete"))
    except BaseException as error:
        native.write_json(args.output / "execution_status.json", dict(status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()

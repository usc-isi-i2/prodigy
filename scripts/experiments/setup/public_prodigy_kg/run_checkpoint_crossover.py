"""Fixed2x2 checkpoint crossover; no alignment, fitting, or selection."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from . import run_native as native

ENCODER = ("layer_list.0.", "layer_list.1.", "initial_input_mlp.")
INFERENCE = ("layer_list.2.", "initial_label_mlp.", "learned_label_embedding.",
             "final_input_mlp.", "final_label_mlp.")
CONDITIONS = ((2000,2000),(2000,8000),(8000,2000),(8000,8000))
EPISODES = 128


def partition(state):
    result = {"encoder": [], "inference": []}
    for key in state:
        if key.startswith(ENCODER):
            result["encoder"].append(key)
        elif key.startswith(INFERENCE) or key == "logit_scale":
            result["inference"].append(key)
        else:
            raise ValueError("Unassigned state key: " + key)
    if not all(result.values()):
        raise ValueError("Both state partitions must be present")
    return result


def assemble(encoder, inference):
    if encoder.keys() != inference.keys():
        raise ValueError("Checkpoint state inventories differ")
    parts = partition(encoder)
    result = {}
    for owner, source in (("encoder",encoder),("inference",inference)):
        for key in parts[owner]:
            if encoder[key].shape != inference[key].shape or encoder[key].dtype != inference[key].dtype:
                raise ValueError("Checkpoint state contract differs: " + key)
            result[key] = source[key].detach().clone()
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for field in ("source","checkpoint-dir","trajectory","upstream","output"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--gpu",type=int,choices=(2,3),default=3)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,key,value.resolve())
    for root in (args.source,args.trajectory):
        if json.loads((root/"execution_status.json").read_text())["status"]!="complete":
            raise ValueError("Completed source and trajectory required")
    records=json.loads((args.source/"paired_episodes/index.json").read_text())["records"]
    if len(records)!=500 or [r["ordinal"] for r in records]!=list(range(500)):
        raise ValueError("Expected500 fresh episodes")
    checkpoints={s:args.checkpoint_dir/f"state_dict_{s}.ckpt" for s in (2000,8000)}
    trajectory_protocol=json.loads((args.trajectory/"protocol.json").read_text())
    index_hash=native.file_sha256(args.source/"paired_episodes/index.json")
    if trajectory_protocol["source_index_sha256"]!=index_hash:
        raise ValueError("Trajectory source differs")
    hashes={str(s):native.file_sha256(path) for s,path in checkpoints.items()}
    if any(trajectory_protocol["checkpoint_sha256"][s]!=h for s,h in hashes.items()):
        raise ValueError("Trajectory checkpoint differs")
    plan=dict(conditions=[list(x) for x in CONDITIONS],episodes=EPISODES,
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        effective_params_sha256=native.file_sha256(args.source/"effective_params.json"),
        source=str(args.source),trajectory=str(args.trajectory),source_index_sha256=index_hash,
        trajectory_summary_sha256=native.file_sha256(args.trajectory/"summary.json"),
        checkpoint_sha256=hashes,upstream=native.verify_upstream(args.upstream),
        ownership=dict(encoder=list(ENCODER),inference=list(INFERENCE)+["logit_scale"]),
        prediction="Early inference improves late encoder and late inference hurts early encoder; both crossovers bad means incompatibility, inconclusive",
        intervention="All conditions clamp preM data rows to corresponding saved trajectory U1 after natural parity1e-4; label rows unchanged",
        fitting=False,parity_atol=1e-4)
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),WANDB_MODE="offline")
    native.runtime_versions()
    native.isolate_native_path(args.upstream)
    os.chdir(args.upstream)
    import torch
    import numpy as np
    from .run_saved_readout import construct_model
    from .run_trajectory import forward_checkpoint
    from .run_anchor_assignment import clamp_pre_data
    from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics
    torch.set_num_threads(1)
    params=json.loads((args.source/"effective_params.json").read_text())
    if (params["layers"],params["batch_size"],params["n_way"],params["n_shots"],params["n_query"])!=("S2,UX,M2",1,20,3,4):
        raise ValueError("Unexpected public model/episode recipe")
    states={s:torch.load(path,map_location="cpu")["model"] for s,path in checkpoints.items()}
    plan["state_keys"]=partition(states[2000])
    trajectory_receipts={(r["step"],r["file"]):r for r in json.loads((args.trajectory/"summary.json").read_text())["receipts"]}
    args.output.mkdir(parents=True,exist_ok=False)
    native.write_json(args.output/"protocol.json",plan)
    rows,receipts=[],[]
    try:
        for encoder,inference in CONDITIONS:
            name=f"E{encoder}I{inference}"
            model=construct_model(params,checkpoints[encoder],"cuda:0")
            state=assemble(states[encoder],states[inference])
            model.load_state_dict(state,strict=True)
            native.verify_imports(args.upstream)
            directory=args.output/name
            directory.mkdir()
            for record in records[:EPISODES]:
                path=args.source/"paired_episodes"/record["file"]
                if path.resolve().parent!=(args.source/"paired_episodes").resolve() or native.file_sha256(path)!=record["sha256"]:
                    raise ValueError("Source receipt mismatch")
                trajectory_file=f"{encoder}/{record['file']}"
                reference_path=args.trajectory/trajectory_file
                receipt=trajectory_receipts[(encoder,trajectory_file)]
                if native.file_sha256(reference_path)!=receipt["sha256"] or receipt["source_sha256"]!=record["sha256"]:
                    raise ValueError("Trajectory receipt mismatch")
                reference=torch.load(reference_path,map_location="cpu")
                artifact=torch.load(path,map_location="cpu")["native_capture"]
                if not all(artifact["state"]["training"].values()):
                    raise ValueError("All native module modes must be training")
                with clamp_pre_data(model,reference["pre_embeddings"]) as audit:
                    yt,logits,u1=forward_checkpoint(model,artifact,"cuda:0")
                if not torch.equal(u1,reference["pre_embeddings"]):
                    raise ValueError("Clamped U1 not exact")
                if not torch.equal(yt,reference["y_true_onehot"]):
                    raise ValueError("Query ordering differs")
                if logits.shape!=(80,20) or not torch.isfinite(logits).all():
                    raise ValueError("Invalid nominated query scores")
                if encoder==inference:
                    torch.testing.assert_close(logits,reference["native_logits"],atol=1e-4,rtol=0)
                truth=yt.argmax(1)
                metrics=episode_metrics(truth.numpy(),logits.numpy())
                losses=torch.nn.functional.cross_entropy(logits,truth,reduction="none")
                examples=[dict(query=i,label=int(truth[i]),prediction=int(logits[i].argmax()),nll=float(losses[i])) for i in range(len(truth))]
                row=dict(condition=name,encoder=encoder,inference=inference,ordinal=record["ordinal"],metrics=metrics,clamp=audit)
                rows.append(row)
                out=directory/record["file"]
                torch.save(dict(source_sha256=record["sha256"],trajectory_sha256=receipt["sha256"],logits=logits,y_true_onehot=yt,examples=examples,row=row),out)
                receipts.append(dict(file=str(out.relative_to(args.output)),sha256=native.file_sha256(out),source_sha256=record["sha256"]))
                if (record["ordinal"]+1)%16==0:
                    print(f"{name} {record['ordinal']+1}/{EPISODES}",flush=True)
            for key,value in model.state_dict().items():
                if not torch.equal(value.cpu(),state[key]):
                    raise ValueError("Assembled checkpoint state changed: "+key)
            del model
        means={f"E{e}I{i}":{m:float(np.mean([r["metrics"][m] for r in rows if (r["encoder"],r["inference"])==(e,i)])) for m in rows[0]["metrics"]} for e,i in CONDITIONS}
        native.write_json(args.output/"summary.json",dict(rows=rows,means=means,receipts=receipts))
        native.write_json(args.output/"execution_status.json",dict(status="complete"))
    except BaseException as error:
        native.write_json(args.output/"execution_status.json",dict(status="failed",error=repr(error)))
        raise


if __name__=="__main__":
    main()

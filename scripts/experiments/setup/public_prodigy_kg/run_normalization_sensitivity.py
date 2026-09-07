"""Fixed BatchNorm protocol sensitivity, not recalibration or repair search."""
import argparse
import json
import os
import subprocess
from pathlib import Path
from . import run_native as native

STEPS=(2000,8000)
EPISODES=128


def normalization_artifact(model, artifact, frozen):
    import torch
    state=dict(artifact["state"])
    state["training"]=dict(state["training"])
    inventory={}
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            inventory[name]=dict(track_running_stats=module.track_running_stats,eps=module.eps,
                momentum=module.momentum,captured_training=state["training"][name])
            if not module.track_running_stats or module.running_mean is None or module.running_var is None:
                raise ValueError("Frozen BatchNorm requires checkpoint running statistics")
            if not torch.isfinite(module.running_mean).all() or not torch.isfinite(module.running_var).all() or (module.running_var<=0).any():
                raise ValueError("Invalid checkpoint running statistics")
            inventory[name].update(running_mean_min=float(module.running_mean.min()),
                running_mean_max=float(module.running_mean.max()),running_var_min=float(module.running_var.min()),
                running_var_max=float(module.running_var.max()),num_batches_tracked=int(module.num_batches_tracked))
            if frozen:
                state["training"][name]=False
    if not inventory:
        raise ValueError("Expected BatchNorm modules")
    return dict(artifact,state=state),inventory


def main():
    p=argparse.ArgumentParser(__doc__)
    for field in ("source","checkpoint-dir","trajectory","upstream","output"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--gpu",type=int,choices=(2,3),default=3)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    for root in (args.source,args.trajectory):
        if json.loads((root/"execution_status.json").read_text())["status"]!="complete":
            raise ValueError("Complete source/trajectory required")
    records=json.loads((args.source/"paired_episodes/index.json").read_text())["records"]
    if len(records)!=500 or [r["ordinal"] for r in records]!=list(range(500)):
        raise ValueError("Complete nominated stream required")
    trajectory=json.loads((args.trajectory/"protocol.json").read_text())
    indexhash=native.file_sha256(args.source/"paired_episodes/index.json")
    if trajectory["source_index_sha256"]!=indexhash:raise ValueError("Trajectory source differs")
    checkpoints={s:args.checkpoint_dir/f"state_dict_{s}.ckpt" for s in STEPS}
    hashes={str(s):native.file_sha256(path) for s,path in checkpoints.items()}
    if any(trajectory["checkpoint_sha256"][s]!=h for s,h in hashes.items()):raise ValueError("Checkpoint differs")
    plan=dict(steps=list(STEPS),episodes=EPISODES,source=str(args.source),trajectory=str(args.trajectory),
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        checkpoint_sha256=hashes,source_index_sha256=indexhash,
        source_params_sha256=native.file_sha256(args.source/"effective_params.json"),
        trajectory_summary_sha256=native.file_sha256(args.trajectory/"summary.json"),
        upstream=native.verify_upstream(args.upstream),
        conditions=["captured_batch_statistics","checkpoint_running_statistics"],
        intervention="Freeze ONLY BatchNorm flags; all other modes/RNG/input identical; own checkpoint buffers",
        no_recalibration=True,no_adaptation=True,no_u1_clamp=True,parity_atol=1e-4,
        readout="Support-centered ridge lambda1 scale1 rowL2 nointercept")
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),WANDB_MODE="offline")
    native.runtime_versions();native.isolate_native_path(args.upstream);os.chdir(args.upstream)
    import torch
    import numpy as np
    from .run_saved_readout import construct_model
    from .run_trajectory import forward_checkpoint
    from .role_centering import predict
    from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics
    torch.set_num_threads(1)
    params=json.loads((args.source/"effective_params.json").read_text())
    if params["dropout"]!=0 or params["text_features_dropout"]!=0:
        raise ValueError("Nominated no-dropout protocol required")
    if (params["batch_size"],params["n_way"],params["n_shots"],params["n_query"])!=(1,20,3,4):
        raise ValueError("Unexpected episode dimensions")
    lookup={(r["step"],r["file"]):r for r in json.loads((args.trajectory/"summary.json").read_text())["receipts"]}
    args.output.mkdir(parents=True,exist_ok=False)
    native.write_json(args.output/"protocol.json",plan)
    rows,receipts=[],[]
    try:
        for step,checkpoint in checkpoints.items():
            model=construct_model(params,checkpoint,"cuda:0");native.verify_imports(args.upstream)
            original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            modes={k:m.training for k,m in model.named_modules()}
            directory=args.output/str(step);directory.mkdir()
            for record in records[:EPISODES]:
                path=args.source/"paired_episodes"/record["file"]
                if path.resolve().parent!=(args.source/"paired_episodes").resolve() or native.file_sha256(path)!=record["sha256"]:
                    raise ValueError("Source receipt differs")
                trajectory_file=f"{step}/{record['file']}"
                refpath=args.trajectory/trajectory_file;receipt=lookup[(step,trajectory_file)]
                if native.file_sha256(refpath)!=receipt["sha256"] or receipt["source_sha256"]!=record["sha256"]:
                    raise ValueError("Trajectory receipt differs")
                reference=torch.load(refpath,map_location="cpu")
                saved=torch.load(path,map_location="cpu");artifact=saved["native_capture"]
                if not all(artifact["state"]["training"].values()):raise ValueError("Expected all captured training modes")
                group=saved["result"]["tasks"][0];s,q=group["support_rows"],group["query_rows"]
                labels=artifact["input"][2]
                predictions,embeddings,metrics={}, {}, {}
                for condition,frozen in (("batch",False),("running",True)):
                    modified,inventory=normalization_artifact(model,artifact,frozen)
                    yt,z,x=forward_checkpoint(model,modified,"cuda:0")
                    if not torch.equal(yt,labels[q]):raise ValueError("Query labels/order differ")
                    ridge=predict(x[s],labels[s],x[q],"support")
                    if not frozen:
                        for actual,expected in ((z,reference["native_logits"]),(x,reference["pre_embeddings"]),(ridge,reference["centered_logits"])):
                            torch.testing.assert_close(actual,expected,atol=1e-4,rtol=0)
                    truth=yt.argmax(1)
                    for decoder,logits in (("native",z),("centered",ridge)):
                        if logits.shape!=(80,20) or not torch.isfinite(logits).all():
                            raise ValueError("Invalid nominated query scores")
                        name=condition+"/"+decoder
                        predictions[name]=logits
                        metrics[name]=episode_metrics(truth.numpy(),logits.numpy())
                    embeddings[condition]=x
                    for key,value in model.state_dict().items():
                        if not torch.equal(value.cpu(),original[key]):raise ValueError("State changed: "+key)
                    if {k:m.training for k,m in model.named_modules()}!=modes:raise ValueError("Modes not restored")
                row=dict(step=step,ordinal=record["ordinal"],metrics=metrics,
                    u1_max_change=float((embeddings["batch"]-embeddings["running"]).abs().max()))
                rows.append(row)
                out=directory/record["file"]
                examples={name:[dict(query=i,label=int(truth[i]),prediction=int(logits[i].argmax()),
                    nll=float(torch.nn.functional.cross_entropy(logits[i:i+1],truth[i:i+1]))) for i in range(len(truth))] for name,logits in predictions.items()}
                torch.save(dict(source_sha256=record["sha256"],trajectory_sha256=receipt["sha256"],
                    y_true_onehot=yt,logits=predictions,u1=embeddings,examples=examples,bn_inventory=inventory,row=row),out)
                receipts.append(dict(file=str(out.relative_to(args.output)),sha256=native.file_sha256(out),source_sha256=record["sha256"]))
                if (record["ordinal"]+1)%16==0:print(f"step{step} {record['ordinal']+1}/{EPISODES}",flush=True)
            del model
        means={str(step):{name:{metric:float(np.mean([r["metrics"][name][metric] for r in rows if r["step"]==step])) for metric in rows[0]["metrics"][name]} for name in rows[0]["metrics"]} for step in STEPS}
        native.write_json(args.output/"summary.json",dict(rows=rows,means=means,receipts=receipts))
        native.write_json(args.output/"execution_status.json",dict(status="complete"))
    except BaseException as error:
        native.write_json(args.output/"execution_status.json",dict(status="failed",error=repr(error)));raise


if __name__=="__main__":main()

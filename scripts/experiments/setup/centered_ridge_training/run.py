"""One matched centered/scaled ridge objective; dry-run by default."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.experiments.setup.encoder_solver_isolation.run import commands as isolation_commands

MODE="ridge_centered_scaled"


def commands(root,output,gpu,steps,seed=0):
    if seed!=0:raise ValueError("Fixed nominated seed0 required")
    result=[]
    for entry in isolation_commands(root,output,gpu,steps,seed):
        if entry["mode"]!="ridge_only":continue
        entry["name"]=entry["name"].removesuffix("ridge_only")+MODE
        entry["mode"]=MODE
        command=entry["command"]
        command[command.index("--prefix")+1]=entry["name"]
        command[command.index("--encoder_solver_objective")+1]=MODE
        result.append(entry)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument("--output",type=Path,default=Path("/dataMeR1/phil/gfm/prodigy-centered-ridge-training/log/centered_ridge_training_smoke"))
    p.add_argument("--gpu",type=int,choices=range(4),required=True)
    p.add_argument("--steps",type=int,choices=(20,2500),default=20)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args();root=args.root.resolve();output=args.output.resolve()
    plan=commands(root,output,args.gpu,args.steps)
    if output.exists():raise FileExistsError("New output required; no implicit resume")
    print(json.dumps(plan,indent=2))
    if not args.execute:return
    output.mkdir(parents=True)
    (output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    (output/"revision.txt").write_text(subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True))
    env=dict(os.environ,WANDB_MODE="offline")
    scales=[]
    for entry in plan:
        with (output/(entry["name"]+".log")).open("w") as stream:
            subprocess.run(entry["command"],cwd=root,stdout=stream,stderr=subprocess.STDOUT,env=env,check=True)
        import torch
        for step in (0,args.steps):
            checkpoint=output/"state"/(entry["name"]+"_isolation_v1")/"checkpoint"/f"state_dict_{step}.ckpt"
            if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
            state=torch.load(checkpoint,map_location="cpu",weights_only=False)["model"]
            scales.append(dict(model=entry["name"],step=step,logit_scale=float(state["logit_scale"]),positive_scale=float(state["logit_scale"].exp())))
    (output/"scale_checkpoints.json").write_text(json.dumps(scales,indent=2)+"\n")
    (output/"DONE.json").write_text(json.dumps(dict(arms=2,steps=args.steps,smoke=args.steps==20,mode=MODE))+"\n")


if __name__=="__main__":main()

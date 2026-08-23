from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from mixture_scaling.config import load_config
from mixture_scaling.model import GraphSAGE
from mixture_scaling.probe_label_budget import BUDGETS, structuralize, support_indices
from mixture_scaling.probe_strict import choose_trial, classifier, embed
from mixture_scaling.strict_data import induced_partition, load_raw, load_split, split_hash
from mixture_scaling.strict_metrics import classification_metrics

def labeled(graph):
    y=graph.data.y.numpy(); keep=y>=0
    return graph.data.x.numpy()[keep], y[keep]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--split-root",required=True)
    p.add_argument("--target",required=True); p.add_argument("--feature-mode",choices=("existing","structural"),required=True)
    p.add_argument("--baseline",choices=("raw","random_sage"),required=True); p.add_argument("--encoder-seed",type=int,required=True)
    p.add_argument("--device",type=int,required=True); p.add_argument("--output",required=True); args=p.parse_args()
    if args.device not in (2,3): raise ValueError("GPUs 2/3 only")
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    config=load_config(args.config); protocol=dict(config["protocol"]); raw=load_raw(config["graphs"][args.target]["path"])
    split=load_split(Path(args.split_root)/f"{args.target}.pt",args.target,int(raw["x"].shape[0]))
    train=induced_partition(args.target,config["graphs"][args.target]["path"],split,"train")
    val=induced_partition(args.target,config["graphs"][args.target]["path"],split,"validation")
    if args.feature_mode=="structural": _,mean,std=structuralize(train); structuralize(val,mean,std)
    if args.baseline=="raw": train_x,train_y=labeled(train); val_x,val_y=labeled(val); model=None
    else:
        torch.manual_seed(args.encoder_seed); dim=int(train.data.x.shape[1]); model=GraphSAGE(dim,256,256,1,0.0).to(torch.device(f"cuda:{args.device}"))
        train_x,train_y=embed(model,train,torch.device(f"cuda:{args.device}")); val_x,val_y=embed(model,val,torch.device(f"cuda:{args.device}"))
    selections={}
    for support_seed in range(10):
        for budget in BUDGETS:
            support=support_indices(train_y,budget,support_seed); trials=[]
            for c in map(float,protocol["probe_c_values"]):
                head=classifier(c,support_seed); head.fit(train_x[support],train_y[support])
                trials.append({"c":c,**classification_metrics(val_y,head.predict(val_x),head.predict_proba(val_x),head.classes_)})
            selections[(support_seed,budget)]=(support,choose_trial(trials,"auc"))
    test=induced_partition(args.target,config["graphs"][args.target]["path"],split,"test")
    if args.feature_mode=="structural": structuralize(test,mean,std)
    test_x,test_y=labeled(test) if model is None else embed(model,test,torch.device(f"cuda:{args.device}"))
    results=[]
    for support_seed in range(10):
        for budget in BUDGETS:
            support,chosen=selections[(support_seed,budget)]; head=classifier(float(chosen["c"]),support_seed)
            head.fit(train_x[support],train_y[support]); results.append({"support_seed":support_seed,"labels_per_class":budget,
                "selected_c":chosen["c"],**classification_metrics(test_y,head.predict(test_x),head.predict_proba(test_x),head.classes_)})
    result={"status":"complete","target":args.target,"target_split_hash":split_hash(split),"baseline":args.baseline,
            "feature_mode":args.feature_mode,"encoder_seed":args.encoder_seed,"support_seeds":list(range(10)),
            "budgets":list(BUDGETS),"validation_labels_used_for_fitting":False,"results":results}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result))
    return 0
if __name__=="__main__": raise SystemExit(main())

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from mixture_scaling.config import load_config
from mixture_scaling.model import GraphSAGE
from mixture_scaling.probe_label_budget import structuralize
from mixture_scaling.strict_data import induced_partition, load_raw, load_split, split_hash
from mixture_scaling.strict_metrics import classification_metrics
from mixture_scaling.train_strict import seed_everything

def arrays(model,head,graph,device,classes):
    model.eval(); head.eval()
    with torch.no_grad(): logits=head(model(graph.data.x.to(device),graph.data.edge_index.to(device)))
    y=graph.data.y.numpy(); keep=y>=0; logits=logits[torch.from_numpy(keep).to(device)]
    prob=torch.softmax(logits,dim=-1).cpu().numpy(); pred=classes[prob.argmax(1)]
    return y[keep],pred,prob

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--split-root",required=True)
    p.add_argument("--target",required=True); p.add_argument("--feature-mode",choices=("existing","structural"),required=True)
    p.add_argument("--seed",type=int,required=True); p.add_argument("--device",type=int,required=True); p.add_argument("--output",required=True)
    args=p.parse_args(); out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    if args.device not in (2,3): raise ValueError("GPUs 2/3 only")
    config=load_config(args.config); raw=load_raw(config["graphs"][args.target]["path"]); split=load_split(Path(args.split_root)/f"{args.target}.pt",args.target,int(raw["x"].shape[0]))
    graphs={part:induced_partition(args.target,config["graphs"][args.target]["path"],split,part) for part in ("train","validation")}
    if args.feature_mode=="structural": _,mean,std=structuralize(graphs["train"]); structuralize(graphs["validation"],mean,std)
    seed_everything(args.seed); device=torch.device(f"cuda:{args.device}"); classes=np.unique(graphs["train"].data.y.numpy()); mapping={int(v):i for i,v in enumerate(classes)}
    model=GraphSAGE(int(graphs["train"].data.x.shape[1]),256,256,1,0.0).to(device); head=nn.Linear(256,len(classes)).to(device)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=1e-3,weight_decay=1e-5)
    y=graphs["train"].data.y.numpy(); keep=y>=0; target=torch.tensor([mapping[int(v)] for v in y[keep]],device=device)
    best_auc=-np.inf; best_epoch=0; best=None; patience=0; start=time.monotonic()
    for epoch in range(1,501):
        model.train(); head.train(); optimizer.zero_grad(set_to_none=True)
        embedding=model(graphs["train"].data.x.to(device),graphs["train"].data.edge_index.to(device)); logits=head(embedding[torch.from_numpy(keep).to(device)])
        loss=torch.nn.functional.cross_entropy(logits,target); loss.backward(); optimizer.step()
        vy,vpred,vprob=arrays(model,head,graphs["validation"],device,classes); metrics=classification_metrics(vy,vpred,vprob,classes); auc=metrics["roc_auc_ovr_macro"]
        if auc>best_auc+1e-5:
            best_auc=auc; best_epoch=epoch; patience=0; best={"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"head":{k:v.detach().cpu().clone() for k,v in head.state_dict().items()}}
        else: patience+=1
        if epoch>=30 and patience>=30: break
    model.load_state_dict(best["model"]); head.load_state_dict(best["head"])
    test=induced_partition(args.target,config["graphs"][args.target]["path"],split,"test")
    if args.feature_mode=="structural": structuralize(test,mean,std)
    ty,tpred,tprob=arrays(model,head,test,device,classes); result={"status":"complete","evaluation":"supervised_graphsage_upper_bound","target":args.target,"target_split_hash":split_hash(split),"feature_mode":args.feature_mode,"seed":args.seed,"best_epoch":best_epoch,"validation_auc":best_auc,"elapsed_seconds":time.monotonic()-start,**classification_metrics(ty,tpred,tprob,classes)}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n"); torch.save(best,out.with_suffix(".pt")); print(json.dumps(result))
    return 0
if __name__=="__main__": raise SystemExit(main())

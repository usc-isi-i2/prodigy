from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config
from .context_views import VIEWS, fixed_view, view_dim
from .lattice import (SOURCE_ORDER, load_shared_graph, make_lp_loader, make_mae_loader,
                      next_batch, update_selection)
from .model import MaskedViewFeatureMLP, ViewMLP
from .node_only_transfer import seed_everything, selected_rows
from .train import configure_sampling_backend


def build_model(view, objective, protocol, device):
    feature_dim = int(protocol["input_dim"])
    width = view_dim(feature_dim, view)
    if objective == "lp":
        model = ViewMLP(width, int(protocol["hidden_dim"]), int(protocol["output_dim"]),
                        float(protocol["dropout"]))
    else:
        model = MaskedViewFeatureMLP(width, feature_dim, int(protocol["hidden_dim"]),
                                     int(protocol["output_dim"]), float(protocol["dropout"]))
    return model.to(device)


def fp_loss(model, batch, view, protocol, generator, device):
    return fp_losses([model], batch, view, protocol, generator, device)[0]


def fp_losses(models, batch, view, protocol, generator, device):
    """Evaluate several models on one sampled batch and one shared feature mask."""
    batch = batch.to(device)
    roots = int(batch.batch_size)
    target = batch.x[:roots].float()
    mask = torch.rand(target.shape, generator=generator).to(device) < float(protocol["mask_rate"])
    empty = ~mask.any(1)
    if empty.any(): mask[empty, 0] = True
    masked_target = target.masked_fill(~mask, 0)
    losses = []
    for model in models:
        corrupted = batch.x.float().clone()
        corrupted[:roots] = torch.where(mask, model.mask_token.expand_as(target), target)
        prediction = model(fixed_view(corrupted, batch.edge_index, view)[:roots])
        prediction = prediction.masked_fill(~mask, 0)
        error = 1 - F.cosine_similarity(prediction, masked_target, dim=-1)
        losses.append(error.clamp_min(0).pow(float(protocol["sce_alpha"])).mean())
    return losses


def lp_loss(model, batch, view, device):
    batch = batch.to(device)
    embedding = model(fixed_view(batch.x.float(), batch.edge_index, view))
    src, dst = batch.edge_label_index
    return F.binary_cross_entropy_with_logits(
        (embedding[src] * embedding[dst]).sum(-1), batch.edge_label.float())


@torch.no_grad()
def validate(model, loaders, view, objective, protocol, seed, device):
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval(); torch.manual_seed(seed + 32452843)
        generator = torch.Generator().manual_seed(seed + 32452843)
        values = []
        for loader in loaders.values():
            losses = []
            for _, batch in zip(range(int(protocol["validation_batches"])), loader):
                loss = lp_loss(model, batch, view, device) if objective == "lp" else fp_loss(
                    model, batch, view, protocol, generator, device)
                losses.append(float(loss))
            if not losses: raise RuntimeError("empty validation loader")
            values.append(np.mean(losses))
        return float(np.mean(values))
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        model.train()


def save(path, model, optimizer, step, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = model.encoder if isinstance(model, MaskedViewFeatureMLP) else model
    torch.save({"model": encoder.state_dict(), "pretrain_model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "metadata": metadata}, path)


def train_one(run_id, source, view, objective, graph, config, device, root, seed):
    protocol = dict(config["protocol"]); run_dir = root / view / objective / run_id
    if (run_dir / "summary.json").is_file(): return json.loads((run_dir / "summary.json").read_text())
    if run_dir.exists(): raise FileExistsError(f"partial run exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    seed_everything(seed); model = build_model(view, objective, protocol, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol.get(
        f"context_mlp_{objective}_learning_rate", protocol["learning_rate"])),
        weight_decay=float(protocol["weight_decay"]))
    maker = make_lp_loader if objective == "lp" else make_mae_loader
    if objective == "lp":
        train_loader, val_loader = maker(graph, protocol, False), maker(graph, protocol, True)
    else:
        train_loader = maker(graph, protocol, False, seed); val_loader = maker(graph, protocol, True, seed)
    metadata = {"run_id":run_id,"sources":[source],"seed":seed,"objective":objective,
                "architecture":"fixed_view_mlp","input_view":view,"fanout":protocol["fanout"],
                "checkpoint_selection":"source_ssl_validation_only","protocol":protocol}
    (run_dir / "metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    iterator=None; best=reference=math.inf; best_step=patience=0; start=time.monotonic(); window=[]
    mask_gen=torch.Generator().manual_seed(seed+49979687); checkpoints=set(map(int,protocol["checkpoint_steps"]))
    for step in range(1,int(protocol["max_steps"])+1):
        batch,iterator=next_batch(train_loader,iterator); optimizer.zero_grad(set_to_none=True)
        loss=lp_loss(model,batch,view,device) if objective=="lp" else fp_loss(model,batch,view,protocol,mask_gen,device)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at {step}")
        loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float(protocol["gradient_clip_norm"]))); optimizer.step(); window.append(float(loss))
        if step in checkpoints: save(run_dir/"checkpoints"/f"step_{step}.pt",model,optimizer,step,metadata)
        if step % int(protocol["validation_interval"]): continue
        value=validate(model,{source:val_loader},view,objective,protocol,seed,device); elapsed=time.monotonic()-start
        row={"step":step,"train_loss":float(np.mean(window)),"validation_loss":value,"elapsed_seconds":elapsed,"updates_per_second":step/elapsed,"gradient_norm":grad}
        with (run_dir/"validation.jsonl").open("a") as handle: handle.write(json.dumps(row)+"\n")
        print(json.dumps({"run_id":run_id,**row}),flush=True); window.clear()
        best,best_step,reference,patience,save_best=update_selection(value,step,best,best_step,reference,patience,float(protocol["min_relative_improvement"]))
        if save_best: save(run_dir/"best.pt",model,optimizer,step,metadata)
        if step>=int(protocol["minimum_steps"]) and patience>=int(protocol["patience_evaluations"]): break
    save(run_dir/"checkpoints"/f"step_{step}.pt",model,optimizer,step,metadata)
    summary={**metadata,"status":"complete","best_step":best_step,"best_validation_loss":best,"final_step":step,"elapsed_seconds":time.monotonic()-start}
    (run_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--view",choices=VIEWS,required=True); p.add_argument("--objective",choices=("fp","lp"),required=True); p.add_argument("--selection",default="full"); p.add_argument("--worker-index",type=int,required=True); p.add_argument("--workers",type=int,default=4); p.add_argument("--device",type=int,required=True); p.add_argument("--output-root",required=True); p.add_argument("--cache-root"); p.add_argument("--seed",type=int,default=0); p.add_argument("--fanout",type=int); a=p.parse_args()
    if a.device not in (0,1,2,3): raise ValueError("only GPUs 0-3")
    config=load_config(a.config)
    if a.fanout is not None:
        if a.fanout < 1: raise ValueError("fanout must be positive")
        config["protocol"]["fanout"] = a.fanout
    configure_sampling_backend(config["protocol"]); root=Path(a.output_root)
    for run_id,(source,) in selected_rows(a.selection)[a.worker_index::a.workers]:
        graph=load_shared_graph(source,config["graphs"][source]["path"],"lp" if a.objective=="lp" else "graphmae",config["protocol"],a.seed,Path(a.cache_root or root/"_cache"))
        result=train_one(run_id,source,a.view,a.objective,graph,config,torch.device(f"cuda:{a.device}"),root,a.seed); print(json.dumps({"completed":run_id,"final_step":result["final_step"]}),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())

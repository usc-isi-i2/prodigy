from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .config import load_config
from .context_mlp_transfer import build_model, fp_losses
from .context_views import VIEWS, fixed_view
from .evaluate_lattice import LP_TARGETS, cached_lp_views, load_pair_module, unwrap
from .lattice import SOURCE_ORDER
from .node_only_transfer import feature_split, singleton_rows
from .train import configure_sampling_backend


def load_model(config, checkpoint, view, objective, device):
    protocol=dict(config["protocol"]); protocol.update(checkpoint["metadata"].get("protocol",{}))
    model=build_model(view,objective,protocol,device)
    state=checkpoint["model"] if objective=="lp" else checkpoint["pretrain_model"]
    model.load_state_dict(state); return model.eval(),protocol


def node_loader(graph,nodes,protocol):
    # NeighborLoader attempts to slice every node-level attribute.  The source
    # artifacts also carry list-valued provenance metadata, which is not a
    # sliceable feature tensor and must not enter the sampled evaluation graph.
    graph = Data(x=graph.x.float(), edge_index=graph.edge_index)
    return NeighborLoader(graph,input_nodes=nodes,num_neighbors=[int(protocol["fanout"])],
                          batch_size=int(protocol.get("eval_batch_size",1024)),shuffle=False,num_workers=0)


def assigned_target_rows(targets, rows, worker_index, workers):
    """Shard target/model cells, then regroup to load each graph once per worker."""
    assigned = {}
    cells = [(target, row) for target in targets for row in rows]
    for target, row in cells[worker_index::workers]:
        assigned.setdefault(target, []).append(row)
    return assigned


@torch.no_grad()
def evaluate_fp(target,a,config,rows,device,root):
    graph=unwrap(config["graphs"][target]["path"]); protocol=config["protocol"]
    nodes=feature_split(int(graph.num_nodes),True,a.seed,float(protocol["node_validation_fraction"]))
    pending=[]
    for run_id,_ in rows:
        output=root/a.view/"fp"/f"{run_id}__to__{target}.json"
        if output.is_file(): continue
        checkpoint=torch.load(Path(a.state_root)/a.view/"fp"/run_id/"best.pt",map_location="cpu",weights_only=False)
        model,p=load_model(config,checkpoint,a.view,"fp",device)
        pending.append((run_id,output,checkpoint,model,p))
    if not pending: return
    reference_protocol=pending[0][4]
    if any(item[4] != reference_protocol for item in pending[1:]):
        raise ValueError("cannot share FP samples across checkpoints with different protocols")
    replicate=[[] for _ in pending]
    for rep in range(int(reference_protocol.get("fp_eval_mask_replicates",10))):
        torch.manual_seed(a.seed+80021+rep); generator=torch.Generator().manual_seed(a.seed+80021+rep)
        losses=[[] for _ in pending]; weights=[]
        for batch in node_loader(graph,nodes,reference_protocol):
            batch_losses=fp_losses([item[3] for item in pending],batch,a.view,reference_protocol,generator,device)
            for values,loss in zip(losses,batch_losses): values.append(float(loss))
            weights.append(int(batch.batch_size))
        for values,model_losses in zip(replicate,losses):
            values.append(float(np.average(model_losses,weights=weights)))
    for (run_id,output,checkpoint,_,_), values in zip(pending,replicate):
        payload={"status":"complete","task":"masked_feature_prediction","target":target,"run_id":run_id,
                 "view":a.view,"sources":checkpoint["metadata"]["sources"],"checkpoint_step":int(checkpoint["step"]),
                 "seed":a.seed,"n_nodes":len(nodes),"mask_replicates":len(values),
                 "scaled_cosine_error_mean":float(np.mean(values)),"scaled_cosine_error_std":float(np.std(values)),
                 "replicate_losses":values}
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2)+"\n")


@torch.no_grad()
def embed_nodes(model,graph,nodes,view,protocol,device):
    table=np.empty((len(nodes),int(protocol["output_dim"])),dtype=np.float32); lookup={int(n):i for i,n in enumerate(nodes.tolist())}
    for batch in node_loader(graph,torch.from_numpy(nodes),protocol):
        batch=batch.to(device); roots=batch.n_id[:batch.batch_size].cpu().numpy()
        encoded=model(fixed_view(batch.x.float(),batch.edge_index,view)[:batch.batch_size]).cpu().numpy()
        table[[lookup[int(n)] for n in roots]]=encoded
    return table


@torch.no_grad()
def evaluate_lp(target,a,config,rows,device,root,pair):
    raw=unwrap(config["graphs"][target]["path"]); background_edges,holdout_edges=cached_lp_views(a.state_root,target,a.seed)
    n_nodes=int(raw.num_nodes); background=pair.Adjacency.from_edge_index(background_edges.numpy(),n_nodes); holdout=pair.Adjacency.from_edge_index(holdout_edges.numpy(),n_nodes)
    rng=np.random.default_rng(a.seed); pairs=pair.build_pair_set(background,holdout,"degree_matched",rng,max_positives=int(config["protocol"].get("lp_eval_positives",2000)),holdout_edge_index=holdout_edges.numpy()); val_mask=pair.split_val_mask(pairs,rng); nodes=pairs.nodes()
    graph=Data(x=raw.x.float(),edge_index=background_edges)
    for run_id,_ in rows:
        output=root/a.view/"lp"/f"{run_id}__to__{target}.json"
        if output.is_file(): continue
        checkpoint=torch.load(Path(a.state_root)/a.view/"lp"/run_id/"best.pt",map_location="cpu",weights_only=False)
        model,p=load_model(config,checkpoint,a.view,"lp",device); table=embed_nodes(model,graph,nodes,a.view,p,device); embeddings=pair.NodeEmbeddings(table,nodes,n_nodes); scores=pair.pair_scores(embeddings,pairs,"cosine"); report=pair.evaluate_scores("fixed_view_mlp_cosine",pairs.label,scores,val_mask).as_dict()
        payload={"status":"complete","task":"static_link_prediction","target":target,"run_id":run_id,"view":a.view,"sources":checkpoint["metadata"]["sources"],"checkpoint_step":int(checkpoint["step"]),"seed":a.seed,"n_pairs":len(pairs),"report":report,"gates":{"holdout_leakage_edges":pair.leakage_check(background,pairs),"endpoint_sensitivity":pair.endpoint_sensitivity(embeddings,pairs),"endpoint_permutation_auc":pair.endpoint_permutation_auc(embeddings,pairs,np.random.default_rng(a.seed+1))}}
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--view",choices=VIEWS,required=True); p.add_argument("--objective",choices=("fp","lp"),required=True); p.add_argument("--worker-index",type=int,required=True); p.add_argument("--workers",type=int,default=4); p.add_argument("--device",type=int,required=True); p.add_argument("--state-root",required=True); p.add_argument("--output-root",required=True); p.add_argument("--prodigy-root",default="/dataMeR1/phil/gfm/prodigy-nm-pairs"); p.add_argument("--seed",type=int,default=0); a=p.parse_args()
    config=load_config(a.config); configure_sampling_backend(config["protocol"]); rows=singleton_rows(); targets=SOURCE_ORDER if a.objective=="fp" else LP_TARGETS; device=torch.device(f"cuda:{a.device}"); pair=load_pair_module(Path(a.prodigy_root)) if a.objective=="lp" else None; root=Path(a.output_root)
    for target, target_rows in assigned_target_rows(targets,rows,a.worker_index,a.workers).items():
        evaluate_fp(target,a,config,target_rows,device,root) if a.objective=="fp" else evaluate_lp(target,a,config,target_rows,device,root,pair)
    return 0
if __name__=="__main__": raise SystemExit(main())

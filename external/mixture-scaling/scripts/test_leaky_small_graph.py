"""Matched small-graph ReLU/LeakyReLU screening experiment."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mixture_scaling.config import load_config
from mixture_scaling.lattice import load_shared_graph
from mixture_scaling.node_only_transfer import build_model,make_direct_lp_loader,lp_loss,validate,seed_everything
from mixture_scaling.prefetch_links import PrefetchLinks
from mixture_scaling.evaluate_lattice import load_pair_module

root=Path('/dataMeR1/phil/gfm/mixture-scaling');out=root/'results/leaky_covid_political_s0_20k';out.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(4);torch.cuda.set_device(1);device=torch.device('cuda:1')
cfg=load_config('configs/node_only_transfer.yaml');p=cfg['protocol'];source='covid_political';steps=20000
cache=Path('/dataMeR1/phil/gfm/mixture-scaling-node-only/state/node_only_transfer/_cache')
g=load_shared_graph(source,cfg['graphs'][source]['path'],'lp',p,0,cache);x=g.data.x.to(device)
seed_everything(0);base=build_model('lp',p,device);models={'relu':base,'leaky_relu':copy.deepcopy(base)};models['leaky_relu'].network[1]=torch.nn.LeakyReLU(.01)
assert all(torch.equal(a,b) for a,b in zip(models['relu'].parameters(),models['leaky_relu'].parameters()))
opts={n:torch.optim.AdamW(m.parameters(),lr=p['node_mlp_lp_learning_rate'],weight_decay=p['weight_decay']) for n,m in models.items()}
train=make_direct_lp_loader(g,p,False,0,x);val=make_direct_lp_loader(g,p,True,0,x)
best={n:float('inf') for n in models};beststep={};beststate={};history=[];started=time.monotonic();sums=dict.fromkeys(models,0.)
with PrefetchLinks({source:train},[source]*steps,device,depth=8,workers=4) as batches:
 for step,(_,batch) in enumerate(batches,1):
  for n,m in models.items():
   m.train();opts[n].zero_grad(set_to_none=True);loss=lp_loss(m,batch,device);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),p['gradient_clip_norm']);opts[n].step();sums[n]+=float(loss)
  if step%250==0:
   history.append({'step':step,'train':{n:v/250 for n,v in sums.items()}});sums=dict.fromkeys(models,0.)
  if step%2000==0:
   values={n:validate(m,g,val,'lp',device,p,0) for n,m in models.items()};history[-1]['validation']=values
   for n,v in values.items():
    if v<best[n]:best[n]=v;beststep[n]=step;beststate[n]=copy.deepcopy(models[n].state_dict())
   print(step,values,round(time.monotonic()-started,1),flush=True)
(out/'history.json').write_text(json.dumps(history))
pair=load_pair_module(Path('/dataMeR1/phil/gfm/prodigy-nm-pairs'));n_nodes=len(x)
bg=pair.Adjacency.from_edge_index(torch.cat((g.train_edges,g.train_edges.flip(0)),1).numpy(),n_nodes);hold=pair.Adjacency.from_edge_index(g.validation_edges.numpy(),n_nodes);rng=np.random.default_rng(0)
pairs=pair.build_pair_set(bg,hold,'degree_matched',rng,max_positives=2000,holdout_edge_index=g.validation_edges.numpy());mask=pair.split_val_mask(pairs,rng);nodes=pairs.nodes();ids=np.random.default_rng(2026).choice(n_nodes,min(n_nodes,4096),replace=False)
result={'graph':source,'seed':0,'steps':steps,'negative_slope':.01,'matched_initial_weights':True,'identical_batches':True,'training_seconds':time.monotonic()-started,'arms':{}}
with torch.no_grad():
 for n,m in models.items():
  m.load_state_dict(beststate[n]);m.eval();table=m(x[nodes].float()).cpu().numpy();emb=pair.NodeEmbeddings(table,nodes,n_nodes);scores=pair.pair_scores(emb,pairs,'cosine');report=pair.evaluate_scores(n,pairs.label,scores,mask).as_dict()
  pre=m.network[0](x[ids].float());h=m.network[1](pre);z=F.normalize(m.network[3](h),dim=1).double();z-=z.mean(0);e=torch.linalg.eigvalsh(z.T@z).clamp_min(0);q=e/e.sum();er=float(torch.exp(-(q[q>0]*q[q>0].log()).sum()))
  result['arms'][n]={'best_validation_bce':best[n],'best_step':beststep[n],'cosine_eval':report,'hidden_zero_fraction':float((h==0).float().mean()),'hidden_never_nonzero_units':int((h.abs().max(0).values<1e-10).sum()),'pre_relu_positive_fraction':float((pre>0).float().mean()),'normalized_output_effective_rank':er,'gates':{'holdout_leakage_edges':pair.leakage_check(bg,pairs),'endpoint_sensitivity':pair.endpoint_sensitivity(emb,pairs)}}
  torch.save({'model':beststate[n],'activation':n,'negative_slope':.01,'step':beststep[n]},out/(n+'_best.pt'))
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

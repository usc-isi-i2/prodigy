"""Matched input-coordinate/PCA retraining pilot on the nine-source mixture."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mixture_scaling.binary_metrics import PredictionWindow, log_values, evaluation_report
from mixture_scaling.evaluation_artifacts import atomic_json, save_scores
from mixture_scaling.node_only_transfer import save_checkpoint, validate_lp_metrics
from mixture_scaling.config import load_config
from mixture_scaling.evaluate_lattice import unwrap
from mixture_scaling.lattice import SOURCE_ORDER, load_shared_graph
from mixture_scaling.model import NodeMLP
from mixture_scaling.node_only_transfer import DirectLinkLoader, lp_loss, seed_everything, validate
from mixture_scaling.node_mlp_ladder import resident_sources, source_schedule
from mixture_scaling.prefetch_links import PrefetchLinks
from mixture_scaling.ladder_tracking import tracked_run, LossWindow

BASE=Path('/dataMeR1/phil/gfm/mixture-scaling')
CACHE=Path('/dataMeR1/phil/gfm/mixture-scaling-node-only/state/node_only_transfer/_cache')

class InputView(nn.Module):
 def __init__(self, method, k, transform):
  super().__init__();self.method=method;self.k=k
  self.register_buffer('mean',transform['mean'])
  if method=='pca':self.register_buffer('projection',transform['pca'][:,:k])
  else:self.register_buffer('indices',transform[method][:k])
  self.encoder=NodeMLP(k,256,256,0)
 def forward(self,x):
  x=x-self.mean
  x=x@self.projection if self.method=='pca' else x[:,self.indices]
  return self.encoder(x)

def fit(args,config):
 samples=[];rank_batches=[]
 for name in SOURCE_ORDER:
  graph=unwrap(config['graphs'][name]['path']);sp=torch.load(CACHE/f'{name}_edge_split_s0.pt',map_location='cpu',weights_only=False);e=sp['train_edges'];rng=np.random.default_rng(771)
  ids=rng.choice(e.shape[1],min(2048,e.shape[1]),replace=False);nodes=e[:,torch.from_numpy(ids)].flatten();samples.append(graph.x[nodes].float().clone())
  # Ranking uses training labels and the previously trained teacher, never evaluation labels.
  pos=e[:,torch.from_numpy(rng.choice(e.shape[1],min(1024,e.shape[1]),replace=False))];n=pos.shape[1];neg=torch.from_numpy(rng.integers(0,graph.x.shape[0],size=(2,n*5)));bad=neg[0]==neg[1]
  while bad.any():neg[1,bad]=torch.from_numpy(rng.integers(0,graph.x.shape[0],size=int(bad.sum())));bad=neg[0]==neg[1]
  pairs=torch.cat([pos,neg],1);rank_batches.append((graph.x[pairs.flatten()].float().clone(),torch.cat([torch.ones(n),torch.zeros(n*5)])))
  del graph,sp,e;print('fit_loaded',name,flush=True)
 # Equal source weights, even if a small source supplies fewer endpoints.
 mean=torch.stack([x.double().mean(0) for x in samples]).mean(0)
 cov=sum((x.double()-mean).T@(x.double()-mean)/len(x) for x in samples)/len(samples)
 ev,vec=torch.linalg.eigh(cov);pca=vec.flip(1).float();mean=mean.float()
 ck=torch.load(BASE/'state/node_mlp_ladder_s0_convergence_fast/lp/ladder_r9/checkpoints/step_900000.pt',map_location='cpu',weights_only=False);teacher=NodeMLP(768,256,256,0).to(args.device);teacher.load_state_dict(ck['model']);teacher.eval()
 for p in teacher.parameters():p.requires_grad_(False)
 scores=[]
 for x,y in rank_batches:
  x=x.to(args.device).requires_grad_(True);y=y.to(args.device);z=teacher(x);n=len(y);l=F.binary_cross_entropy_with_logits((z[:n]*z[n:]).sum(1),y);grad=torch.autograd.grad(l,x)[0];scores.append((grad*(x-mean.to(args.device))).abs().mean(0).detach().cpu())
 score=torch.stack(scores).mean(0)
 transform={'mean':mean,'pca':pca,'selected':score.argsort(descending=True),'random':torch.from_numpy(np.random.default_rng(772).permutation(768)),'full':torch.arange(768),'saliency':score,'eigenvalues':ev.flip(0).float(),'teacher_step':ck['step']}
 path=args.root/'transforms.pt';assert not path.exists();torch.save(transform,path);print('FIT_COMPLETE',flush=True)

def run(args,config):
 transform=torch.load(args.root/'transforms.pt',map_location='cpu',weights_only=False)
 protocol=dict(config['protocol'],validation_batches=20)
 graphs={}
 for s in SOURCE_ORDER:
  graphs[s]=load_shared_graph(s,config['graphs'][s]['path'],'lp',protocol,0,CACHE);graphs[s].data.edge_index=torch.empty((2,0),dtype=torch.long);print('loaded',s,flush=True)
 free,_=torch.cuda.mem_get_info();sizes={s:g.data.x.numel()*g.data.x.element_size() for s,g in graphs.items()};resident=resident_sources(sizes,max(0,min(65*2**30,free-10*2**30)))
 features={s:g.data.x.to(args.device) if s in resident else g.data.x for s,g in graphs.items()};train={};val={};check={}
 for s,g in graphs.items():
  # Seeded disjoint halves of the pre-existing held-out edge partition.
  order=torch.randperm(g.validation_edges.shape[1],generator=torch.Generator().manual_seed(773));cut=len(order)//2;assert cut>0
  train[s]=DirectLinkLoader(features[s],g.train_edges,1024,5,True,49979687)
  val[s]=DirectLinkLoader(features[s],g.validation_edges[:,order[:cut]],1024,5,False,32452843)
  check[s]=DirectLinkLoader(features[s],g.validation_edges[:,order[cut:]],1024,5,False,32452844)
 dims=[768] if args.method=='full' else [16,64,256];limit=args.steps_per_source*9
 with ExitStack() as stack:
  arms=[]
  for k in dims:
   seed_everything(0);model=InputView(args.method,k,transform).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=0.0005,weight_decay=1e-5)
   run_id=f'{args.method}_{k}_s0';path=args.root/run_id;path.mkdir(exist_ok=False)
   meta={'run_id':run_id,'sources':list(SOURCE_ORDER),'method':args.method,'input_dim':k,'seed':0,'steps_per_source':args.steps_per_source,'total_steps':limit,'pilot':True,'centering':'source-balanced train-endpoint mean for all arms','selection':'teacher gradient-times-centered-input on training pairs; fixed-budget terminal teacher (900000 updates), not validation-selected','check_split':'second seeded half of canonical held-out edges','parameter_count':sum(p.numel() for p in model.parameters())}
   (path/'metadata.json').write_text(json.dumps(meta,indent=2));wr,log=stack.enter_context(tracked_run(path,meta,args));arms.append({'model':model,'opt':opt,'path':path,'log':log,'wr':wr,'window':LossWindow(),'predictions':PredictionWindow(),'best':float('inf'),'best_step':0,'meta':meta})
  batches=stack.enter_context(PrefetchLinks(train,source_schedule(SOURCE_ORDER,limit),args.device,depth=8,workers=4));start=time.monotonic()
  for step,(source,batch) in enumerate(batches,1):
   for a in arms:
    model=a['model'];a['opt'].zero_grad(set_to_none=True);l,logits=lp_loss(model,batch,args.device,return_logits=True)
    if not torch.isfinite(l):raise FloatingPointError('nonfinite training loss')
    l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);a['opt'].step();a['window'].add(source,float(l.detach()),batch.edge_label.numel())
    a['predictions'].add(source,batch.edge_label,logits)
    if step%250==0 or step==limit:a['log'](step,{**a['window'].flush(),**a['predictions'].flush()})
   if step%2500==0:print('PROGRESS',step,'elapsed',time.monotonic()-start,flush=True)
   if step%18000==0 or step==limit:
    for a in arms:
     reports={s:validate_lp_metrics(a['model'],val[s],args.device,20) for s in SOURCE_ORDER}
     losses={s:r['bce'] for s,r in reports.items()};value=sum(losses.values())/9
     a['log'](step,{'validation/loss':value,**{f'validation/source/{s}/loss':v for s,v in losses.items()},**{key:v for s,r in reports.items() for key,v in log_values(r,f'validation/source/{s}').items()}})
     atomic_json(a['path']/'validation'/f'step_{step}.json',reports)
     if value<a['best']:
      a['best']=value;a['best_step']=step;save_checkpoint(a['path']/'best.pt',a['model'],a['opt'],step,{**a['meta'],'best_step':step,'best_validation_bce':value})
     save_checkpoint(a['path']/'latest.pt',a['model'],a['opt'],step,{**a['meta'],'best_step':a['best_step'],'best_validation_bce':a['best']})
     save_checkpoint(a['path']/'checkpoints'/f'step_{step}.pt',a['model'],a['opt'],step,a['meta'])
     print('VALIDATION',a['meta']['run_id'],step,value,flush=True)
  for a in arms:
   save_checkpoint(a['path']/'terminal.pt',a['model'],a['opt'],limit,{**a['meta'],'best_step':a['best_step']})
   a['model'].load_state_dict(torch.load(a['path']/'best.pt',map_location=args.device,weights_only=False)['model'])
   metrics={}
   for source in SOURCE_ORDER:
    _,vy,vs=validate_lp_metrics(a['model'],val[source],args.device,20,return_data=True)
    _,ty,ts=validate_lp_metrics(a['model'],check[source],args.device,20,return_data=True)
    y=np.concatenate([vy,ty]);scores=np.concatenate([vs,ts]);mask=np.arange(len(y))<len(vy)
    metrics[source]=evaluation_report(y,scores,mask)
    save_scores(a['path']/'check'/f'{source}.scores.npz',labels=y,dot_logits=scores,validation_mask=mask)
    atomic_json(a['path']/'check'/f'{source}.json',metrics[source])
    a['log'](a['best_step'],log_values(metrics[source]['test'],f'check/source/{source}'))
   losses={s:r['test']['bce'] for s,r in metrics.items()}
   result={**a['meta'],'status':'complete','best_step':a['best_step'],'validation_bce':a['best'],'check_bce':sum(losses.values())/9,'check_by_source':losses,'check_metrics':metrics,'elapsed_seconds':time.monotonic()-start}
   a['wr'].summary.update(result);(a['path']/'summary.json').write_text(json.dumps(result,indent=2));print('COMPLETE',json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['fit','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--device',type=int,default=0,choices=[0,1,2,3]);p.add_argument('--method',choices=['selected','random','pca','full']);p.add_argument('--steps-per-source',type=int,default=10000);args=p.parse_args();torch.set_num_threads(4);torch.cuda.set_device(args.device);args.device=torch.device(f'cuda:{args.device}');args.root.mkdir(parents=True,exist_ok=True)
 args.wandb_project='node-mlp-input-study';args.wandb_mode='offline';args.wandb_group=args.root.name;args.log_interval=250
 config=load_config('configs/node_only_transfer.yaml')
 fit(args,config) if args.phase=='fit' else run(args,config)
if __name__=='__main__':main()

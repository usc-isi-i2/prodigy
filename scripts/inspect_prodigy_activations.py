import sys,json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'/dataMeR1/phil/gfm/prodigy-role-topology')
import torch
import torch.nn.functional as F
from scripts.experiments.setup.target_performance_mechanisms.run_episode_cardinality import make_model
from scripts.experiments.setup.target_performance_mechanisms.replay import trace_stages,clone_batch

torch.set_num_threads(4)
root=Path('/dataMeR1/phil/gfm/prodigy-mechanisms-complement/log/target_mechanisms/mixture_complementarity_20260906_v2')
rec=next(r for r in json.loads((root/'checkpoint_inventory.json').read_text())['models'] if r['model_id']=='nmloo_without_covid')
state=torch.load(rec['checkpoint'],map_location='cpu',weights_only=True)['model']
protocol=json.loads((root/'original/protocol.json').read_text());protocol['device']='0'
protocol['catalog']='/dataMeR1/phil/gfm/prodigy-role-topology/docs/graph_catalog.json';protocol['config']='/dataMeR1/phil/gfm/prodigy-role-topology/scripts/experiments/setup/final_core/training.yaml'
def stats(x):
 x=x.double();x=x-x.mean(0);v=torch.linalg.eigvalsh(x.T@x/max(1,len(x)-1)).clamp_min(0).flip(0);p=v/v.sum();return {'rank':float(torch.exp(-(p[p>0]*p[p>0].log()).sum())),'top1':float(p[0]),'top10':float(p[:10].sum())}
out={'checkpoint':rec,'batch_count':8,'targets':{}}
with torch.no_grad():
 for target in ['covid_political','election2020','facebook_page_reference','twibot20','ukr_rus_suspended']:
  directory=root/'original'/target;first=torch.load(directory/'batches/batch_000.pt',map_location='cpu',weights_only=False)
  cache=json.loads((directory/'cache.json').read_text());model=make_model(protocol,target,cache['graph_path'],first[0].x.shape[1],state)
  stages=defaultdict(list);relu=defaultdict(list);handles=[]
  for name,module in model.named_modules():
   if isinstance(module,torch.nn.ReLU):
    handles.append(module.register_forward_hook(lambda m,a,y,key=name:relu[key].append(y.detach().reshape(-1,y.shape[-1])[:4096].clone())))
  for i in range(8):
   batch=clone_batch(torch.load(directory/f'batches/batch_{i:03d}.pt',map_location='cpu',weights_only=False))
   with trace_stages(model,batch[0]) as trace:model(*batch)
   for name,x in trace.items():stages[name].append(x.cpu())
  for h in handles:h.remove()
  result={}
  for name,parts in stages.items():
   x=torch.cat(parts);result[name]={'n':len(x),'zero_fraction':float((x==0).float().mean()),'never_active':int((x.abs().max(0).values<1e-10).sum()),'raw':stats(x),'normalized':stats(F.normalize(x,dim=1))}
  out['targets'][target]={'stages':result,'relu':{name:{'zero_fraction':float((torch.cat(parts)==0).float().mean()),'never_active':int((torch.cat(parts).abs().max(0).values<1e-10).sum())} for name,parts in relu.items()}}
  print(target,json.dumps(result),flush=True)
Path('prodigy_activation_results.json').write_text(json.dumps(out,indent=2))

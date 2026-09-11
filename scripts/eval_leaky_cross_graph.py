"""Paired cross-graph evaluation of the small activation comparison."""
import json
from pathlib import Path
import numpy as np
import torch
from mixture_scaling.config import load_config
from mixture_scaling.evaluate_lattice import load_pair_module,unwrap,cached_lp_views
from mixture_scaling.model import NodeMLP
root=Path('/dataMeR1/phil/gfm/mixture-scaling');out=root/'results/leaky_covid_political_s0_20k'
torch.set_num_threads(4);torch.cuda.set_device(1);device=torch.device('cuda:1')
cfg=load_config('configs/node_only_transfer.yaml');pair=load_pair_module(Path('/dataMeR1/phil/gfm/prodigy-nm-pairs'))
models={}
for arm in ['relu','leaky_relu']:
 c=torch.load(out/(arm+'_best.pt'),map_location='cpu',weights_only=False);m=NodeMLP(768,256,256,0)
 if arm=='leaky_relu':m.network[1]=torch.nn.LeakyReLU(c['negative_slope'])
 m.load_state_dict(c['model']);models[arm]=m.to(device).eval()
results={}
with torch.no_grad():
 for target,entry in cfg['graphs'].items():
  g=unwrap(entry['path']);n=int(g.num_nodes)
  be,he=cached_lp_views(root/'state/node_mlp_ladder_s0_convergence_recovery',target,0)
  bg=pair.Adjacency.from_edge_index(be.numpy(),n);ho=pair.Adjacency.from_edge_index(he.numpy(),n);rng=np.random.default_rng(0)
  pairs=pair.build_pair_set(bg,ho,'degree_matched',rng,max_positives=2000,holdout_edge_index=he.numpy());mask=pair.split_val_mask(pairs,rng);nodes=pairs.nodes();x=g.x[nodes].float().to(device);del g
  results[target]={}
  for arm,m in models.items():
   emb=pair.NodeEmbeddings(m(x).cpu().numpy(),nodes,n);scores=pair.pair_scores(emb,pairs,'cosine');report=pair.evaluate_scores(arm,pairs.label,scores,mask).as_dict()
   gates={'holdout_leakage_edges':pair.leakage_check(bg,pairs),'endpoint_sensitivity':pair.endpoint_sensitivity(emb,pairs)};assert gates['holdout_leakage_edges']==0 and gates['endpoint_sensitivity']>0
   results[target][arm]={'report':report,'gates':gates}
  print(target,results[target]['relu']['report']['auc'],results[target]['leaky_relu']['report']['auc'],flush=True)
  del x,bg,ho,be,he
(out/'cross_graph.json').write_text(json.dumps(results,indent=2))

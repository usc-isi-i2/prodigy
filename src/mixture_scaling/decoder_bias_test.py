"""Matched one-graph decoder ablation using the established disjoint LP trainer."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from . import mini_lp as lp
from .model import NodeMLP
from .config import load_config
from .binary_metrics import binary_report
from .evaluation_artifacts import atomic_json, save_scores

REFERENCE=Path('/dataMeR1/phil/gfm/mixture-scaling-disjoint-neighbor/state/disjoint_context_s0')
PAIRS=Path('/dataMeR1/phil/gfm/mixture-scaling-uniform-eval/state/uniform_eval_s0_v2/node_neighbors_disjoint_uniform')
class DecoderMLP(NodeMLP):
    def __init__(self, arm):
        super().__init__(1536,256,256,0)
        self.arm=arm
        if arm!='raw': self.decoder_bias=torch.nn.Parameter(torch.tensor(-math.log(5.)))
        if arm=='affine': self.log_scale=torch.nn.Parameter(torch.tensor(0.))
    def transform(self, logits):
        if self.arm=='affine': logits=logits*self.log_scale.exp()
        if self.arm!='raw': logits=logits+self.decoder_bias
        return logits

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',choices=['raw','bias','affine'],required=True);p.add_argument('--device',type=int,choices=range(4),required=True)
    args=p.parse_args();args.root=str(Path(args.root)/args.arm)
    args.view='node_neighbors';args.seed=0;args.max_steps=100000;args.validation_interval=2000;args.patience=3;args.log_interval=100
    args.wandb_mode='offline';args.wandb_project='nonzero-mini-transfer';args.wandb_group='midterm-decoder-bias-test'
    torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(f'cuda:{args.device}');source='midterm'
    config=load_config('configs/nonzero_mini_transfer.yaml');config['protocol']['decoder_ablation']=dict(arm=args.arm,bias_initial=-math.log(5.) if args.arm!='raw' else None,scale_initial=1.,source=source)
    original_prepare=lp.prepare;original_features=lp.view_features;original_score=lp.score
    lp.prepare=lambda source,config,root,seed:original_prepare(source,config,REFERENCE,seed)
    lp.view_features=lambda source,data,root,view,seed,device:original_features(source,data,REFERENCE,view,seed,device)
    lp.model_for=lambda view,device:DecoderMLP(args.arm).to(device)
    lp.score=lambda model,x,pairs:model.transform(original_score(model,x,pairs))
    lp.train(source,config,args,device)
    run=Path(args.root)/args.view/'lp'/('ss_'+source)
    checkpoint=torch.load(run/'best.pt',map_location=device,weights_only=False)
    model=DecoderMLP(args.arm).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    data=lp.prepare(source,config,Path(args.root),0);x=lp.view_features(source,data,Path(args.root),args.view,0,device)
    reference=PAIRS/f'ss_{source}__to__{source}.scores.npz';z=np.load(reference)
    pairs=torch.tensor(np.stack([z['u'],z['v']]),device=device)
    with torch.no_grad(): logits=torch.cat([lp.score(model,x,pairs[:,i:i+4096]) for i in range(0,pairs.shape[1],4096)]).cpu().numpy()
    mask=~z['validation_mask'];report=binary_report(z['labels'][mask],logits[mask])
    result=dict(arm=args.arm,source=source,checkpoint_step=checkpoint['step'],test=report,bias=float(model.decoder_bias.detach()) if args.arm!='raw' else 0.,scale=float(model.log_scale.exp().detach()) if args.arm=='affine' else 1.,pair_reference=str(reference),baseline_bce=-(math.log(1/6)/6+5*math.log(5/6)/6))
    atomic_json(run/'decoder_test.json',result);save_scores(run/'decoder_test.scores.npz',labels=z['labels'],logits=logits,validation_mask=z['validation_mask'])
    print(result,flush=True)
if __name__=='__main__':main()

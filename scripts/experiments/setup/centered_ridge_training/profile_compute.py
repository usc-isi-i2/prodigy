"""No-update forward/backward microbenchmark with a checked exact U1 early exit."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from experiments.run_shared_graph import write_json
from models.layer_classes import MetagraphLayer
from .evaluate import build_model
from scripts.experiments.setup.target_performance_mechanisms.replay import clone_batch,batch_hash


class StopAtU1(Exception):pass


@contextmanager
def early_exit(model):
    """The existing forward already computed ridge loss immediately before M."""
    modules=[m for m in model.layer_list if isinstance(m,MetagraphLayer)]
    if len(modules)!=1:raise ValueError('Expected single metagraph')
    def stop(module,args,kwargs):
        if model.encoder_solver_ridge_loss is None:raise ValueError('No live U1 objective before metagraph')
        raise StopAtU1()
    handle=modules[0].register_forward_pre_hook(stop,with_kwargs=True)
    try:yield
    finally:handle.remove()


def backward(model,batch,mode):
    old=model.params.get('encoder_solver_objective','native')
    active=model.encoder_solver_training
    model.params['encoder_solver_objective']='native' if mode=='native' else 'ridge_centered_scaled'
    model.encoder_solver_training=True
    try:
        if mode=='early_exit':
            with early_exit(model):
                try:model(*batch)
                except StopAtU1:pass
                else:raise ValueError('Early exit did not fire')
            loss=model.encoder_solver_ridge_loss
        else:
            yt,yp,_=model(*batch)
            loss=F.cross_entropy(yp,yt) if mode=='native' else model.encoder_solver_ridge_loss
        loss.backward()
        return loss.detach()
    finally:
        model.params['encoder_solver_objective']=old;model.encoder_solver_training=active


def main():
    p=argparse.ArgumentParser(__doc__)
    for field in ('batch','checkpoint','params','output'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--input-kind',choices=('actual_training','target_proxy'),required=True)
    p.add_argument('--training-receipt',type=Path,help='Required for actual_training; JSON must attest actual_training and batch_sha256')
    p.add_argument('--gpu',type=int,choices=range(4),default=3)
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not 1<=args.warmup<=10 or not 5<=args.repeats<=50:raise ValueError('Bounded warmup/repeats required')
    plan=dict(input_kind=args.input_kind,batch=str(args.batch),checkpoint=str(args.checkpoint),
        modes=['native','compatibility','early_exit'],warmup=args.warmup,repeats=args.repeats,
        scope='Forward/backward only; no optimizer steps, dataloading, graph sampling, checkpointing or end-to-end throughput',
        timing='CUDA synchronized wall time; clone/reset excluded identically for all modes',
        limitation='Target-cache proxy does not estimate 30way training throughput' if args.input_kind=='target_proxy' else 'Single cached training batch microbenchmark')
    print(json.dumps(plan,indent=2))
    if not args.execute:return
    torch.set_num_threads(4);device=f'cuda:{args.gpu}';torch.cuda.set_device(args.gpu)
    batch=torch.load(args.batch,map_location='cpu',weights_only=False)
    if args.input_kind=='actual_training':
        if args.training_receipt is None:raise ValueError('Actual-training provenance required')
        receipt=json.loads(args.training_receipt.read_text())
        if receipt.get('actual_training') is not True or receipt.get('batch_sha256')!=batch_hash(batch):raise ValueError('Training receipt differs')
    params=json.loads(args.params.read_text()) if args.params.suffix=='.json' else __import__('yaml').safe_load(args.params.read_text())
    params=params.get('params',params)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model']
    model=build_model(params,state,device).train()
    gpu_state={k:v.clone() for k,v in model.state_dict().items()}
    plan.update(batch_hash=batch_hash(batch),file_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.batch,args.checkpoint,args.params)})
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'protocol.json',plan)
    def reset():
        model.load_state_dict(gpu_state,strict=True);model.zero_grad(set_to_none=True)
        model.train();torch.manual_seed(710009);return clone_batch(batch,device)
    gradients={};losses={}
    for mode in ('compatibility','early_exit'):
        losses[mode]=float(backward(model,reset(),mode))
        gradients[mode]={k:None if v.grad is None else v.grad.detach().cpu().clone() for k,v in model.named_parameters()}
    torch.testing.assert_close(torch.tensor(losses['compatibility']),torch.tensor(losses['early_exit']),atol=1e-5,rtol=0)
    errors={}
    for key,a in gradients['compatibility'].items():
        b=gradients['early_exit'][key]
        if (a is None)!=(b is None):raise ValueError('Gradient inventory differs: '+key)
        if a is not None:
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
            errors[key]=float((a-b).abs().max())
    result={}
    for mode in ('native','compatibility','early_exit'):
        for _ in range(args.warmup):backward(model,reset(),mode)
        timings=[];peaks=[]
        for _ in range(args.repeats):
            current=reset();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            baseline=torch.cuda.memory_allocated();start=time.perf_counter()
            backward(model,current,mode);torch.cuda.synchronize()
            timings.append(time.perf_counter()-start);peaks.append(dict(total=torch.cuda.max_memory_allocated(),increment=torch.cuda.max_memory_allocated()-baseline))
        result[mode]=dict(seconds=timings,median_seconds=float(__import__('statistics').median(timings)),memory_bytes=peaks)
    reset()
    write_json(args.output/'summary.json',dict(results=result,loss_parity=losses,gradient_max_abs_errors=errors,
        input_kind=args.input_kind,no_optimizer_steps=True,gpu=torch.cuda.get_device_name(args.gpu)))
    write_json(args.output/'execution_status.json',dict(status='complete'))


if __name__=='__main__':main()

"""Persistent shared-graph, target-major NM evaluation of selected campaign models."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from scripts.experiments.setup.nm_interventions_overnight.plan import TARGETS, HOLDOUT


def discover(run_dirs):
    jobs=[]
    for run in run_dirs:
        manifest=json.loads((Path(run)/'manifest.json').read_text())
        for path in sorted(Path(run).glob('job_*/result.json')):
            result=json.loads(path.read_text())
            if result.get('status')!='complete':
                continue
            params=json.loads((path.parent/'effective_config.json').read_text())
            params['campaign_revision']=manifest['revision']
            if params['exp_name'].startswith('smoke_'):
                continue
            state=Path(result['checkpoint_dir']).parent
            selection=json.loads((state/'selection.json').read_text())
            if selection['status']!='complete':
                continue
            jobs.append(dict(params=params,selection=selection,state=str(state)))
    keys=[j['params']['prefix'] for j in jobs]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate completed model IDs; explicitly resolve retry provenance')
    return jobs


def model_from_params(params, checkpoint, device):
    import torch
    from experiments.layers import get_module_list
    from models.general_gnn import SingleLayerGeneralGNN
    p=dict(params,device=device)
    layers=get_module_list(p['layers'],p['emb_dim'],edge_attr_dim=None,input_dim=p['input_dim'],
        dropout=p['dropout'],reset_after_layer=p['reset_after_layer'],
        attention_mask_scheme=p['attention_mask_scheme'],has_final_back=p['has_final_back'],
        msg_pos_only=p.get('meta_gnn_pos_only',False),batch_norm_metagraph=not p['no_bn_metagraph'],
        batch_norm_encoder=not p['no_bn_encoder'],encoder_gnn_type=p['gnn_type'],gnn_use_relu=False)
    model=SingleLayerGeneralGNN(torch.nn.ModuleList(layers),
        initial_label_mlp=torch.nn.Linear(768,p['emb_dim']),params=p,
        text_dropout=torch.nn.Dropout(p['text_features_dropout']))
    payload=torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval()


def worker(dataset,jobs,output,worker_id,episodes,targets):
    os.setsid()
    import torch
    from experiments.nm_campaign import atomic_json,materialize,evaluate
    output=Path(output)
    stream=(output/f'worker_{worker_id}.log').open('a',buffering=1)
    os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
    torch.set_num_threads(2)
    torch.autograd.set_detect_anomaly(False)
    device=torch.device('cuda:0')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    try:
        for target in targets:
            pending=[]
            for job in jobs:
                path=output/'cells'/job['params']['prefix']/f'{target}.json'
                if path.exists():
                    old=json.loads(path.read_text())
                    if old['checkpoint']!=job['selection']['checkpoint'] or old['episodes']!=episodes or old['protocol']!='nmi_fixed_nm_v1':
                        raise ValueError(f'Resume mismatch: {path}')
                else:
                    pending.append(job)
            if not pending:
                continue
            print(f'TARGET {target}: {len(pending)} models; materializing {episodes} episodes',flush=True)
            batches,fingerprint=materialize(dataset,pending[0]['params'],target,'test',episodes)
            for job in pending:
                p=job['params'];selection=job['selection'];checkpoint=selection['checkpoint']
                model=model_from_params(p,checkpoint,device)
                metrics=evaluate(model,batches,device)
                with open(checkpoint,'rb') as f:
                    sha=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()
                payload=dict(protocol='nmi_fixed_nm_v1',model_id=p['prefix'],target=target,
                    sources=selection['sources'],holdout=HOLDOUT,seed=p['seed'],
                    flags=selection['flags'],checkpoint=checkpoint,checkpoint_sha256=sha,
                    checkpoint_step=selection['best_step'],training_steps=selection['training_steps'],
                    training_revision=p.get('campaign_revision','recorded in training manifest'),
                    evaluation_revision=revision,fingerprint=fingerprint,**metrics)
                atomic_json(output/'cells'/p['prefix']/f'{target}.json',payload)
                print(f"DONE {p['prefix']} {target} auc={metrics['roc_auc']:.6f} seconds={metrics['seconds']:.1f}",flush=True)
                del model
                gc.collect()
            del batches
            gc.collect()
        atomic_json(output/f'worker_{worker_id}_status.json',dict(status='complete'))
    except BaseException:
        atomic_json(output/f'worker_{worker_id}_status.json',dict(status='failed',error=traceback.format_exc()))
        traceback.print_exc()
        raise


def main():
    a=argparse.ArgumentParser();a.add_argument('--run-dirs',nargs='+',required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3],choices=range(4))
    a.add_argument('--workers-per-gpu',type=int,default=2);a.add_argument('--episodes',type=int,default=512)
    a.add_argument('--targets',nargs='+',default=list(TARGETS));a.add_argument('--dry-run',action='store_true')
    args=a.parse_args()
    jobs=discover(args.run_dirs)
    if not jobs: raise ValueError('No completed production models')
    print(f'{len(jobs)} models x {len(args.targets)} targets = {len(jobs)*len(args.targets)} cells',flush=True)
    if args.dry_run:return
    import torch
    from experiments.params import get_params
    from experiments.run_single_experiment import load_dataset
    from experiments.run_shared_graph import prepare_shared_dataset,start_on_gpu
    from experiments.nm_campaign import atomic_json
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/'plan.json',dict(models=[j['params']['prefix'] for j in jobs],targets=args.targets,episodes=args.episodes))
    p=get_params(['--config',jobs[0]['params']['config']])
    torch.set_num_threads(4);torch.autograd.set_detect_anomaly(False)
    dataset=prepare_shared_dataset(load_dataset(p))
    slots=[g for g in args.gpus for _ in range(args.workers_per_gpu)]
    context=torch.multiprocessing.get_context('spawn')
    processes=[]
    for i,gpu in enumerate(slots):
        selected=jobs[i::len(slots)]
        if not selected:continue
        process=context.Process(target=worker,args=(dataset,selected,str(args.output),i,args.episodes,args.targets))
        start_on_gpu(process,gpu);processes.append(process)
    for process in processes:process.join()
    failures=[p.exitcode for p in processes if p.exitcode!=0]
    atomic_json(args.output/'status.json',dict(status='failed' if failures else 'complete',failures=failures))
    if failures:raise RuntimeError(f'Worker failures: {failures}')

if __name__=='__main__':main()
